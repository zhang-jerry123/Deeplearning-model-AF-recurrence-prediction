from pathlib import Path
from numpy.lib import row_stack
from torch.utils.data import Dataset, DataLoader
import utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import scipy.io as sio
from typing import Optional
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
import os

invalid_data = {10230, 10222}


class EcgDataSet(Dataset):
    def __init__(self, num_leads, ecg_path: Path, clinical_data: pd.DataFrame, output_type: utils.PredictionType,
                 is_training: bool):
        self.output_type = output_type
        self.clinical_data = clinical_data
        self.rows = list(clinical_data.index)
        if is_training:
            clinical_data_resampled = utils.resample_minor_class(
                clinical_data, output_type.value)
            self.rows = list(clinical_data_resampled.index)
        self.ecg_path = ecg_path
        self.data = []
        for row in tqdm(self.rows):
            ecg = utils.EcgData.from_mat(sio.loadmat(
                str(self.ecg_path / f'{row}.mat')), self.clinical_data)
            ecg_segments = ecg.preprocessing(num_leads)
            self.data.extend(ecg_segments)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        return [entry.ecg.astype(np.float32), entry.clinical_data[utils.feature_columns].to_numpy(dtype=np.float32)], utils.recurrence_label_map[entry.clinical_data[f'{self.output_type.value}']].astype(np.float32)


class ECGDataModule(pl.LightningDataModule):
    def __init__(self, ecg_path: Path, meta_data_path: Path, output_type: utils.PredictionType, leads: int,
                ecg_freq: Optional[int] = None,
                 use_k_fold: bool = False,
                 k: int = 0, num_fold: int = 5, random_seed=32):

        super().__init__()
        self.ecg_path = ecg_path
        self.meta_data_path = meta_data_path
        self.output_type = output_type
        assert leads in [8, 11]
        self.leads = leads

        self.use_k_fold = use_k_fold
        self.k = k
        self.num_fold = num_fold
        self.random_seed = random_seed
        self.ecg_freq = ecg_freq

    def prepare_data(self):
        self.clinical_data = pd.read_csv(self.meta_data_path, index_col=1)
        if self.output_type == utils.PredictionType.AF:
            self.clinical_data = self.clinical_data[self.clinical_data.index.isin(
                invalid_data) == False]
        if self.output_type == utils.PredictionType.REC:
            self.clinical_data = self.clinical_data[
                (self.clinical_data.Recurrence.isna() == False) & (
                    self.clinical_data.index.isin(invalid_data) == False)]
        print(self.clinical_data)
        self.clinical_data = utils.preprocessing_clinical_data(
            self.clinical_data)
        if self.ecg_freq is not None:
            print(self.ecg_freq)
            self.clinical_data = self.clinical_data[self.clinical_data.frequency == self.ecg_freq]

    def setup(self, stage: Optional[str] = None):
        if not self.use_k_fold:
            train_rows, val_rows = train_test_split(self.clinical_data,
                                                    stratify=self.clinical_data[f'{self.output_type.value}'],
                                                    test_size=0.25,
                                                    shuffle=True)

        else:
            # choose fold to train on
            kf = StratifiedKFold(n_splits=self.num_fold,
                                 shuffle=True, random_state=self.random_seed)
            all_splits = [k for k in kf.split(
                self.clinical_data, self.clinical_data[f'{self.output_type.value}_label'])]
            train_indexes, val_indexes = all_splits[self.k]
            train_rows, val_rows = self.clinical_data.iloc[
                train_indexes], self.clinical_data.iloc[val_indexes]
        self.ecg_train = EcgDataSet(
            self.leads, self.ecg_path, train_rows, self.output_type, False)
        self.ecg_val = EcgDataSet(
            self.leads, self.ecg_path, val_rows, self.output_type, False)

    def train_dataloader(self):
        return DataLoader(self.ecg_train, batch_size=256, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.ecg_val, batch_size=32, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.ecg_val, batch_size=32, num_workers=os.cpu_count())
