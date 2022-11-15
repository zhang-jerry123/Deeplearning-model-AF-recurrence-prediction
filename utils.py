from dataclasses import dataclass
import numpy as np
import pandas as pd
import scipy.signal as signal
from torch_ecg._preprocessors.base import preprocess_multi_lead_signal
import enum
from pathlib import Path
from sklearn.utils import resample
from sklearn import preprocessing
from typing import List
from sklearn.preprocessing import MinMaxScaler

useful_leads = [
    'I',
    'II',
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6'
]

twelve_leads = [
    'I',
    'II',
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6',
    'A1',
    'A2',
    'A3'
]


class PredictionType(enum.Enum):
    AF = 'clinicalAFtype'
    REC = 'Recurrence'


@dataclass
class EcgData:
    number: int
    leads: List[str]
    ecg: np.ndarray
    frequency: int
    clinical_data: pd.Series

    @staticmethod
    def from_mat(mat: dict, clinical_data: pd.DataFrame):
        data = mat['data'][0]
        ecg = data['ecg'][0].astype(np.float32)
        number = int(data['number'][0][0])
        frequency = data['frequency'][0][0][0]
        labels = data['labels'][0]
        if len(labels) == 1:
            labels = [label[0] for label in labels[0]]
        else:
            labels = [label[0][0] for label in labels]
        for i, label in enumerate(labels):
            if label.startswith('xECG0'):
                labels[i] = f"A{label[-1]}"
        return EcgData(number, labels, ecg, frequency, clinical_data.loc[number])

    def down_sample(self):
        if self.frequency == 500:
            return self
        assert self.frequency == 2000
        ecg = signal.upfirdn([1], x=self.ecg, down=int(
            self.frequency / 500), axis=0) * 2
        return EcgData(self.number, self.leads, ecg, 500, self.clinical_data)

    def band_pass(self):
        ecg = preprocess_multi_lead_signal(self.ecg.T, self.frequency, band_fs=[
                                           0.05, 100], filter_type='fir')
        return EcgData(self.number, self.leads, ecg.transpose(), self.frequency, self.clinical_data)

    def filter_leads(self, num_leads: int):
        assert num_leads == 8 or num_leads == 11
        idx = []
        leads = useful_leads if num_leads == 8 else twelve_leads
        for lead in leads:
            try:
                lead_idx = self.leads.index(lead)
                idx.append(lead_idx)
            except ValueError:
                print(f'Lead {lead} not found in {self.number}, {self.leads}')

        return EcgData(self.number, leads, self.ecg[:, idx], self.frequency, self.clinical_data)

    def split_to_segments(self, segment_size: int):
        ecg_len = self.ecg.shape[0]
        remaining = ecg_len % segment_size
        num = ecg_len // segment_size
        ecg_to_segment = self.ecg[:ecg_len - remaining]
        segments = ecg_to_segment.reshape(num, segment_size, self.ecg.shape[1])
        return [EcgData(self.number, self.leads, segments[[segment_idx]], self.frequency, self.clinical_data) for
                segment_idx in range(
                segments.shape[0])]

    def segment(self, length: int = 5120):
        return EcgData(self.number, self.leads, self.ecg[:length], self.frequency, self.clinical_data)

    def preprocessing(self, num_leads: int = 8, segment_size: int = 5120):
        return self.filter_leads(num_leads).down_sample().band_pass().split_to_segments(segment_size)


def resample_minor_class(df: pd.DataFrame, column: str):
    df_description = df[column].describe()
    major_label = df_description['top']
    major_count = df_description['freq']
    minor_rows = df[df[column] != major_label]
    major_rows = df[df[column] == major_label]
    minor_resampled = resample(
        minor_rows, n_samples=major_count, random_state=42)
    concatenated_rows = pd.concat([minor_resampled, major_rows])
    return concatenated_rows.sample(frac=1)

SUFFIX = '_0NO_1Yes'

feature_columns = ['ScoresCHADS_CHF', 'ScoresCHADS_Hypertension',
                   'ScoresCHADS_DiabetesMellitus', 'ScoresCHADS_Stroke',
                   'ScoresCHADS_VascularDisease', 'gender_label', 'Height', 'Weight', 'BMI', 'CHADS', 'Age']


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def preprocessing_clinical_data(clinical_data: pd.DataFrame):
    clinical_data = clinical_data.drop(columns=['Unnamed: 0'])
    for idx, column in enumerate(clinical_data.columns):
        if column.endswith(SUFFIX):
            clinical_data[column].replace('NO', 0, inplace=True)
            clinical_data[column].replace('YES', 1, inplace=True)
            clinical_data[column].astype('int')
            clinical_data.rename(
                columns={column: remove_suffix(column, SUFFIX)}, inplace=True)

    af_le = preprocessing.LabelEncoder()
    recurrence_le = preprocessing.LabelEncoder()
    gender_le = preprocessing.LabelEncoder()
    gender_le.fit(clinical_data['Ma_Gender_0Male_1female'])
    af_le.fit(clinical_data[PredictionType.AF.value])
    recurrence_le.fit(clinical_data[PredictionType.REC.value])
    clinical_data['gender_label'] = gender_le.transform(
        clinical_data['Ma_Gender_0Male_1female'])
    height_norm = MinMaxScaler()
    weight_norm = MinMaxScaler()
    bmi_norm = MinMaxScaler()
    chadsvasc_norm = MinMaxScaler()
    age_norm = MinMaxScaler()
    clinical_data.Ma_Height_cm.fillna(
        clinical_data.Ma_Height_cm.mean(), inplace=True)
    clinical_data.Ma_Weight_kg.fillna(
        clinical_data.Ma_Weight_kg.mean(), inplace=True)
    clinical_data.MZ_BMI.fillna(clinical_data.MZ_BMI.mean(), inplace=True)
    clinical_data['Height'] = height_norm.fit_transform(
        clinical_data[['Ma_Height_cm']])
    clinical_data['Weight'] = weight_norm.fit_transform(
        clinical_data[['Ma_Weight_kg']])
    clinical_data['BMI'] = bmi_norm.fit_transform(clinical_data[['MZ_BMI']])
    clinical_data['CHADS'] = chadsvasc_norm.fit_transform(
        clinical_data[['MZ_CHADSVASc']])
    clinical_data['Age'] = age_norm.fit_transform(clinical_data[['MZ_Age']])
    clinical_data[f'{PredictionType.AF.value}_label'] = af_le.transform(
        clinical_data[PredictionType.AF.value])
    clinical_data[f'{PredictionType.REC.value}_label'] = recurrence_le.transform(
        clinical_data[PredictionType.REC.value])
    return clinical_data


def summurise(result):
    metrics = {}
    for fold in result:
        for key, value in fold[0].items():
            metric_value = metrics.get(key, [])
            metric_value.append(value)
            metrics[key] = metric_value
    result = {}
    for key, value in metrics.items():
        result[key] = f'{np.mean(value):.2f}±{np.std(value):.2f}'
    return result

