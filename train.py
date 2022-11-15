from models import ECGModel
from dataset import ECGDataModule
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings
import torch
import utils
import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


DATA_PATH = Path('data/')
ECG_PATH = DATA_PATH / 'ecgs/'
META_PATH = DATA_PATH / 'clinical_data.csv'

if __name__ == '__main__':
    results = []
    nums_folds = 5
    random_seed = 32
    leads = 8
    prediction_type = utils.PredictionType.REC
    experiment_time = datetime.datetime.now().strftime("%m%d-%H:%M")
    use_clinical = False
    ecg_freq = 500
    experiment_name = f'{prediction_type.value}-{500}-{leads}-clinical:{use_clinical}-{experiment_time}/'
    log_dir = Path('tb_logs')
    experiment_log_path =  log_dir / experiment_name
    experiment_log_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path('checkpoints') / experiment_name
    result_path = Path('results') / experiment_name
    
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    
    for k in range(nums_folds):
        print(f'training fold {k}')
        checkpoint_path_fold = checkpoint_path / f'fold_{k}'
        checkpoint_path_fold.mkdir(exist_ok=True, parents=True)
        data_module = ECGDataModule(ECG_PATH, META_PATH, prediction_type, leads, use_k_fold=True, k=k,
                                    num_fold=nums_folds, random_seed=random_seed, ecg_freq=ecg_freq)
        model = ECGModel(n_leads=leads, dropout_prob=0.25, use_clinical_feat=use_clinical)
        logger = TensorBoardLogger(log_dir, name=experiment_name,
                                   sub_dir=f'fold_{k}')
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path_fold, save_top_k=1, monitor="val_auroc", mode="max")
        trainer = Trainer(max_epochs=500, 
                          logger=logger,
                          accelerator='auto',
                          log_every_n_steps=1,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='min'),
                                     TQDMProgressBar(),
                                     checkpoint_callback])
        trainer.fit(model, data_module)
        result = trainer.test(ckpt_path="best", datamodule=data_module)
        print('='*20)
        print(f'Result of fold {k}')
        print(result)
        results.append(result)
        result_path.mkdir(parents=True, exist_ok=True)
        torch.save(result, result_path / f'fold_{k}.pt')
    print(results)
    print(utils.summurise(results))
