from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAUROC, BinaryROC
from torch import nn
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class ClinicalModel(nn.Module):
    def __init__(self, int_features):
        super().__init__()
        self.fc1 = nn.Linear(int_features, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BottleNeckBlock(nn.Module):

    def __init__(self, ch1: int, ch2: int, kernel_size: int, p):
        super().__init__()
        self.skip_connection = nn.Sequential(
            nn.Conv2d(ch1, ch2, 1),
            nn.MaxPool2d((2, 1))
        )
        self.left_branch1 = nn.Sequential(
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            nn.Conv2d(ch1, ch2, kernel_size=(kernel_size, 1),
                      stride=(1, 1), padding=(p, 0))
        )
        self.left_branch2 = nn.Sequential(
            nn.BatchNorm2d(ch2),
            nn.ReLU(),
            nn.Conv2d(ch2, ch2, kernel_size=(kernel_size, 1),
                      stride=(2, 1), padding=(p, 0))
        )

    def forward(self, x):
        shortcut = self.skip_connection(x)
        x = self.left_branch1(x)
        x = self.left_branch2(x)
        return x + shortcut


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class EcgNetwork2D(nn.Module):
    def __init__(self, n_leads: int = 8, dropout_prob: float = 0, use_clinical_feat: bool = False):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.bn_1 = nn.Sequential(
            BottleNeckBlock(1, 16, 7, 3),
            BottleNeckBlock(16, 16, 7, 3),
            BottleNeckBlock(16, 16, 7, 3),
            nn.Dropout(self.dropout_prob)
        )
        self.bn_2 = nn.Sequential(
            BottleNeckBlock(16, 32, 5, 2),
            BottleNeckBlock(32, 32, 5, 2),
            BottleNeckBlock(32, 32, 5, 2),
            nn.Dropout(self.dropout_prob)
        )
        self.bn_3 = nn.Sequential(
            BottleNeckBlock(32, 64, 3, 1),
            BottleNeckBlock(64, 64, 3, 1),
            BottleNeckBlock(64, 64, 3, 1),
            nn.Dropout(self.dropout_prob)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, n_leads)),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.channel_conv = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=(1, n_leads)),
        #     # nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )

        # self.fc = nn.Linear(5131, 1)
        self.use_clinical_feat = use_clinical_feat
        fc_in = 1280 + (11 if self.use_clinical_feat else 0)
        self.fc = nn.Linear(fc_in, 1)
        self.softmax = nn.Softmax(dim=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        ecg = x[0]
        clinical = x[1]
        ecg = self.bn_1(ecg)
        ecg = self.bn_2(ecg)
        ecg = self.bn_3(ecg)
        ecg = self.channel_conv(ecg)
        ecg = F.dropout(ecg, self.dropout_prob)
        if self.use_clinical_feat:
            x = torch.concat(
                [ecg.view(-1, num_flat_features(ecg)), clinical], dim=1)
            x = self.fc(x)
        else:
            x = self.fc(ecg.view(-1, num_flat_features(ecg)))
        return self.act(x)

class EcgWrapper(LightningModule):
    def __init__(self, model, n_leads: int = 8, dropout_prob: float = 0, use_clinical_feat: bool = False):
        super().__init__()
        self.model = model

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()

        self.test_accuracy = BinaryAccuracy()
        self.test_auroc = BinaryAUROC()
        self.test_f1 = BinaryF1Score()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_roc = BinaryROC(thresholds=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x).view(-1)
        loss = F.binary_cross_entropy(logits, y.float())
        self.train_accuracy.update(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(-1)
        loss = F.binary_cross_entropy(logits, y.float())
        self.val_accuracy.update(logits, y)
        self.val_f1.update(logits, y)
        self.val_recall.update(logits, y)
        self.val_precision.update(logits, y)
        self.val_auroc.update(logits, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(-1)
        loss = F.binary_cross_entropy(logits, y.float())
        self.test_accuracy.update(logits, y)
        self.test_f1.update(logits, y)
        self.test_precision.update(logits, y)
        self.test_auroc.update(logits, y)
        self.test_recall.update(logits, y)
        self.test_roc.update(logits, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        self.log("test_auroc", self.test_auroc, prog_bar=True)
        self.log("test_precision", self.test_precision, prog_bar=True)
        self.log("test_recall", self.test_recall, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-06)


class ECGModel(LightningModule):
    def __init__(self, n_leads: int = 8, dropout_prob: float = 0, use_clinical_feat: bool = False):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.bn_1 = nn.Sequential(
            BottleNeckBlock(1, 16, 7, 3),
            BottleNeckBlock(16, 16, 7, 3),
            BottleNeckBlock(16, 16, 7, 3),
            nn.Dropout(self.dropout_prob)
        )
        self.bn_2 = nn.Sequential(
            BottleNeckBlock(16, 32, 5, 2),
            BottleNeckBlock(32, 32, 5, 2),
            BottleNeckBlock(32, 32, 5, 2),
            nn.Dropout(self.dropout_prob)
        )
        self.bn_3 = nn.Sequential(
            BottleNeckBlock(32, 64, 3, 1),
            BottleNeckBlock(64, 64, 3, 1),
            BottleNeckBlock(64, 64, 3, 1),
            nn.Dropout(self.dropout_prob)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, n_leads)),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.channel_conv = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=(1, n_leads)),
        #     # nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )

        # self.fc = nn.Linear(5131, 1)
        self.use_clinical_feat = use_clinical_feat
        fc_in = 1280 + (11 if self.use_clinical_feat else 0)
        self.fc = nn.Linear(fc_in, 1)
        self.softmax = nn.Softmax(dim=1)
        self.act = nn.Sigmoid()

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()

        self.test_accuracy = BinaryAccuracy()
        self.test_auroc = BinaryAUROC()
        self.test_f1 = BinaryF1Score()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_roc = BinaryROC(thresholds=10)
        self.loss_weights = torch.tensor([0.2, 0.8])

    def forward(self, x):
        ecg = x[0]
        clinical = x[1]
        ecg = self.bn_1(ecg)
        ecg = self.bn_2(ecg)
        ecg = self.bn_3(ecg)
        ecg = self.channel_conv(ecg)
        ecg = F.dropout(ecg, self.dropout_prob)
        if self.use_clinical_feat:
            x = torch.concat(
                [ecg.view(-1, num_flat_features(ecg)), clinical], dim=1)
            x = self.fc(x)
        else:
            x = self.fc(ecg.view(-1, num_flat_features(ecg)))
        return self.act(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x).view(-1)
        loss = F.binary_cross_entropy(logits, y.float())
        self.train_accuracy.update(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(-1)
        loss = F.binary_cross_entropy(logits, y.float())
        self.val_accuracy.update(logits, y)
        self.val_f1.update(logits, y)
        self.val_recall.update(logits, y)
        self.val_precision.update(logits, y)
        self.val_auroc.update(logits, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(-1)
        loss = F.binary_cross_entropy(logits, y.float())
        self.test_accuracy.update(logits, y)
        self.test_f1.update(logits, y)
        self.test_precision.update(logits, y)
        self.test_auroc.update(logits, y)
        self.test_recall.update(logits, y)
        self.test_roc.update(logits, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        self.log("test_auroc", self.test_auroc, prog_bar=True)
        self.log("test_precision", self.test_precision, prog_bar=True)
        self.log("test_recall", self.test_recall, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-06)


class BottleNeckBlock1D(nn.Module):
    def __init__(self, ch1: int, ch2: int, kernel_size: int, padding: int, leads: int):
        super().__init__()
        self.skip_connection = nn.Sequential(
            nn.Conv1d(ch1 * leads, ch2 * leads, groups=ch1 * leads, kernel_size=1),
            nn.MaxPool1d(2)
        )
        self.left_branch1 = nn.Sequential(
            nn.BatchNorm1d(ch1 * leads),
            nn.ReLU(),
            nn.Conv1d(ch1 * leads, ch2 * leads, groups=ch1 * leads, kernel_size=kernel_size, padding=padding, stride=1)
        )
        self.left_branch2 = nn.Sequential(
            nn.BatchNorm1d(ch2 * leads),
            nn.ReLU(),
            nn.Conv1d(ch2 * leads, ch2 * leads, groups=1, kernel_size=kernel_size, padding=padding, stride=2)
        )

    def forward(self, x):
        shortcut = self.skip_connection(x)
        x = self.left_branch1(x)
        x = self.left_branch2(x)
        return x + shortcut


class EcgNetwork1D(nn.Module):
    def __init__(self, n_leads: int = 8, dropout_prob: float = 0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.n_lead = n_leads
        self.bn_1 = nn.Sequential(
            BottleNeckBlock1D(1, 16, 7, 3, self.n_lead),
            BottleNeckBlock1D(16, 16, 7, 3, self.n_lead),
            nn.Dropout(self.dropout_prob)
        )
        self.bn_2 = nn.Sequential(
            BottleNeckBlock1D(16, 32, 5, 2, self.n_lead),
            BottleNeckBlock1D(32, 32, 5, 2, self.n_lead),
            nn.Dropout(self.dropout_prob)
        )
        self.bn_3 = nn.Sequential(
            BottleNeckBlock1D(32, 64, 3, 1, self.n_lead),
            BottleNeckBlock1D(64, 64, 3, 1, self.n_lead),
            nn.Dropout(self.dropout_prob)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv1d(80, 1, kernel_size=64),
            nn.ReLU()
        )
        self.fc = nn.Linear(449, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = x[0]
        x = x.view(x.shape[0], x.shape[3], x.shape[2])
        x = self.bn_1(x)
        x = self.bn_2(x)
        x = self.bn_3(x)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = self.channel_conv(x)
        x = F.dropout(x, self.dropout_prob)
        x = x.view(-1, num_flat_features(x))
        x = self.fc(x)
        return self.act(x)

