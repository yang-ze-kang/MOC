from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth


class Generic_Omic_Survival_Dataset(Dataset):

    def __init__(self,
                 dataset_dir='',
                 study='',
                 target_gene=None,
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 n_bins=4,
                 ignore=[],
                 patient_strat=False,
                 label_col='survival_months',
                 filter_dict={},
                 eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.dataset_dir = dataset_dir
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.split_path = os.path.join(dataset_dir, study, '5fold-rna')
        clinical = pd.read_csv(os.path.join(dataset_dir, study, 'raw_data',
                                            'clinical.csv'), low_memory=False)

        if target_gene != None:
            gene = pd.read_csv(os.path.join(
                dataset_dir, study, 'raw_data', 'rnaseq.csv'))
            if target_gene == 'mrna_rnaseq':
                target_gene = set(pd.read_csv(os.path.join(dataset_dir,
                                                           'mrna.csv'))['gene_name'])
                target_gene = np.concatenate(
                    [np.array(list(target_gene), dtype=object) + mode for mode in ['', '_rnaseq']])
                target_gene = set(target_gene)
            elif target_gene == 'signatures_rnaseq':
                target_gene = set(pd.read_csv(os.path.join(dataset_dir, 'signatures.csv'),
                                              dtype=str,
                                              delimiter=',').stack())
                target_gene = np.concatenate(
                    [np.array(list(target_gene), dtype=object) + mode for mode in ['', '_rnaseq']])
                target_gene = set(target_gene)
            elif target_gene == 'signatures_rnaseq_cnv':
                target_gene = set(pd.read_csv(os.path.join(dataset_dir, 'signatures.csv'),
                                              dtype=str,
                                              delimiter=',').stack())
                target_gene = np.concatenate(
                    [np.array(list(target_gene), dtype=object) + mode for mode in ['', '_rnaseq', '_cnv']])
                target_gene = set(target_gene)
            elif target_gene != None:
                raise NotImplementedError
            target_gene.add('case_id')
            target_gene = list(set(gene.columns) & target_gene)
            target_gene = sorted(target_gene)
            print('gene num: ', len(target_gene)-1)
            gene = gene[target_gene]
            slide_data = pd.merge(clinical, gene)
        else:
            target_gene = set()
            slide_data = clinical
        assert label_col in slide_data.columns
        self.label_col = label_col

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col],
                                      q=n_bins,
                                      retbins=True,
                                      labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        disc_labels, q_bins = pd.cut(patients_df[label_col],
                                     bins=q_bins,
                                     retbins=True,
                                     labels=False,
                                     right=False,
                                     include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient: slide_ids})

        self.patient_dict = patient_dict

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {
            'case_id': patients_df['case_id'].values,
            'label': patients_df['label'].values
        }

        self.slide_data = slide_data
        metadata = ['disc_label', 'case_id', 'label', 'slide_id', 'age',
                    'survival_months', 'censorship', 'gender'
                    ]
        s1 = set(metadata) | set(target_gene)
        s2 = set(self.slide_data.columns)
        assert s1 == s2
        self.metadata = metadata

        for col in slide_data.drop(self.metadata, axis=1).columns:
            if not pd.Series(col).str.contains('|_cnv|_rnaseq|_rna|_mut')[0]:
                print(col)

        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        r"""

        """
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(
                self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        r"""

        """
        patients = np.unique(np.array(
            self.slide_data['case_id']))  # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] ==
                                        p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]]  # get patient label
            patient_labels.append(label)

        self.patient_data = {
            'case_id': patients,
            'label': np.array(patient_labels)
        }

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        r"""

        """

        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n',
              self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' %
                  (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' %
                  (i, self.slide_cls_ids[i].shape[0]))

    def get_split_from_df(self,
                          all_splits: dict,
                          contrast=False,
                          split_key: str = 'train',
                          scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['case_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            if contrast:
                split = Generic_Contrast_Split(df_slice,
                                               metadata=self.metadata,
                                               data_mode=self.data_mode,
                                               signatures=self.signatures,
                                               data_dir=self.data_dir,
                                               label_col=self.label_col,
                                               patient_dict=self.patient_dict,
                                               num_classes=self.num_classes)
            else:
                split = Generic_Split(df_slice,
                                      metadata=self.metadata,
                                      data_mode=self.data_mode,
                                      signatures=self.signatures,
                                      data_dir=self.data_dir,
                                      label_col=self.label_col,
                                      patient_dict=self.patient_dict,
                                      num_classes=self.num_classes)
        else:
            split = None

        return split

    def return_splits(self, from_id: bool = True, csv_path: str = None, contrast=False):
        if from_id:
            raise NotImplementedError
        else:
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, contrast=contrast,
                                                 split_key='train')
            if contrast:
                train_val_split = self.get_split_from_df(all_splits=all_splits,split_key='train')
            else:
                train_val_split = None
            val_split = self.get_split_from_df(all_splits=all_splits,
                                               split_key='val')
            # self.get_split_from_df(all_splits=all_splits, split_key='test')
            test_split = None

            # --> Normalizing Data
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            # test_split.apply_scaler(scalers=scalers)
            # <--
        return train_split, train_val_split, val_split  # , test_split

    def get_list(self, ids):
        return self.slide_data['slide_filename'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None


class Generic_Muti_Survival_Dataset(Generic_Omic_Survival_Dataset):

    def __init__(self, data_dir, data_mode: str = 'omic', **kwargs):
        super(Generic_Muti_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.data_mode = data_mode
        self.use_h5 = False
        if 'sigsets' in self.data_mode:
            self.signatures = pd.read_csv(
                os.path.join(self.dataset_dir, 'signatures.csv'))
        else:
            self.signatures = None

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def get_item_by_case_id(self, case_id):
        idx = self.slide_data[self.slide_data['case_id'] == case_id].index.to_list()[
            0]
        return self[idx]

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = torch.LongTensor([self.slide_data['disc_label'][idx]])
        event_time = torch.Tensor([self.slide_data[self.label_col][idx]])
        c = torch.Tensor([self.slide_data['censorship'][idx]])
        slide_ids = self.patient_dict[case_id]

        if not self.use_h5:
            # genomic data
            if 'sigsets' in self.data_mode:
                omic1 = torch.tensor(
                    self.genomic_features[self.omic_names[0]].iloc[idx]).type(torch.FloatTensor)
                omic2 = torch.tensor(
                    self.genomic_features[self.omic_names[1]].iloc[idx]).type(torch.FloatTensor)
                omic3 = torch.tensor(
                    self.genomic_features[self.omic_names[2]].iloc[idx]).type(torch.FloatTensor)
                omic4 = torch.tensor(
                    self.genomic_features[self.omic_names[3]].iloc[idx]).type(torch.FloatTensor)
                omic5 = torch.tensor(
                    self.genomic_features[self.omic_names[4]].iloc[idx]).type(torch.FloatTensor)
                omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx]).type(torch.FloatTensor)
                omic_features = {
                    'x_omic1': omic1,
                    'x_omic2': omic2,
                    'x_omic3': omic3,
                    'x_omic4': omic4,
                    'x_omic5': omic5,
                    'x_omic6': omic6
                }
            elif 'omic' in self.data_mode:
                omic_features = {'x_omic': torch.tensor(self.genomic_features.iloc[idx]).type(torch.FloatTensor)}
            else:
                omic_features = {'x_omic': torch.ones((1, 1)).type(torch.FloatTensor)}
            # WSI data
            if 'path' in self.data_mode:
                path_features = []
                for slide_id in slide_ids:
                    wsi_path = os.path.join(
                        self.data_dir, 'pt_files',
                        '{}.pt'.format(slide_id.rstrip('.svs')))
                    wsi_bag = torch.load(wsi_path)
                    path_features.append(wsi_bag)
                path_features = torch.cat(path_features, dim=0)
            else:
                path_features = torch.ones((1, 256, 256))
            # return
            if self.data_mode in ['path', 'path_omic', 'path_omic_sigsets', 'omic', 'omic_sigsets']:
                return (path_features, omic_features, label, event_time, c)
            else:
                raise NotImplementedError('Mode [%s] not implemented.' %
                                          self.data_mode)


class Generic_Split(Generic_Muti_Survival_Dataset):

    def __init__(self,
                 slide_data,
                 metadata,
                 data_mode,
                 signatures=None,
                 data_dir=None,
                 label_col=None,
                 patient_dict=None,
                 num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.data_mode = data_mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        # --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        if self.genomic_features.shape[1] == 0:
            self.target_gene = 0

        self.signatures = signatures

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate(
                    [omic + mode for mode in ['', '_mut', '_cnv', '_rnaseq']])
                omic = sorted(
                    series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        print("Shape", self.genomic_features.shape)
        # <--

    def __len__(self):
        return len(self.slide_data)

    # --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        if self.genomic_features.shape[1] != 0:
            scaler_omic = StandardScaler().fit(self.genomic_features)
            return (scaler_omic, )
        else:
            return None

    # <--

    # --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple = None):
        if self.genomic_features.shape[1] != 0:
            transformed = pd.DataFrame(
                scalers[0].transform(self.genomic_features))
            transformed.columns = self.genomic_features.columns
            self.genomic_features = transformed

    # <--

    def set_split_id(self, split_id):
        self.split_id = split_id


class Generic_Contrast_Split(Generic_Muti_Survival_Dataset):

    def __init__(self,
                 slide_data,
                 metadata,
                 data_mode,
                 signatures=None,
                 data_dir=None,
                 label_col=None,
                 patient_dict=None,
                 num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.data_mode = data_mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.training = True
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        # --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        if self.genomic_features.shape[1] == 0:
            self.target_gene = 0

        self.signatures = signatures

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate(
                    [omic + mode for mode in ['', '_mut', '_cnv', '_rnaseq']])
                omic = sorted(
                    series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        print("Shape", self.genomic_features.shape)
        indexs = self.slide_data['survival_months'].argsort()[::-1]
        pairs = []
        for i, index in enumerate(indexs):
            if self.slide_data.loc[index, 'censorship'] == 1:
                continue
            self.slide_data.loc[index, 'order'] = i+1
            for index2 in indexs[i+1:]:
                if self.slide_data.loc[index2, 'censorship'] != 1:
                    pairs.append([index, index2])
        self.pairs = pairs
    
    def get_num(self):
        return len(self.slide_data)

    def __len__(self):
        if self.training:
            return len(self.pairs)
        else:
            return super().__len__()

    def __getitem__(self, index):
        if self.training:
            d1 = super().__getitem__(self.pairs[index][0])
            d2 = super().__getitem__(self.pairs[index][1])
            return ((d1,self.slide_data.loc[self.pairs[index][0],'order']),(d2,self.slide_data.loc[self.pairs[index][1],'order']))
        else:
            return super().__getitem__(index)

    # --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        if self.genomic_features.shape[1] != 0:
            scaler_omic = StandardScaler().fit(self.genomic_features)
            return (scaler_omic, )
        else:
            return None

    # <--

    # --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple = None):
        if self.genomic_features.shape[1] != 0:
            transformed = pd.DataFrame(
                scalers[0].transform(self.genomic_features))
            transformed.columns = self.genomic_features.columns
            self.genomic_features = transformed

    # <--

    def set_split_id(self, split_id):
        self.split_id = split_id