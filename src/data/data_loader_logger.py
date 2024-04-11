# -*- coding: utf-8 -*-
from sklearn import datasets
import pandas as pd
import numpy as np

import torch, os
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LTRLoggerDataLoader(Dataset):
    '''
    Child class of the full-information dataloader. 
    Used to train a logging policy, which is trained on a fraction of queries.
    '''
    def __dcg_norm(self, label_vector, k):
        '''
        get the dcg normalizing factor to calculate ndcg@k
        '''
        label_vector = np.array(label_vector)
        label_vector = sorted(label_vector, reverse=True)
        weights_per_rank = 1/np.log2(np.array(range(len(label_vector))) +2)
        label_vector *= weights_per_rank
        if np.sum(label_vector[:k]) > 0.:
            return np.sum(label_vector[:k])
        else:
            return 1.

    def __init__(self, train_df, k, train_frac):
        self.uniq_qids = np.unique(train_df['qid'].tolist())
        self.qid_map = dict({k:v for k, v in zip(sorted(self.uniq_qids), list(range(len(self.uniq_qids))))})
        self._max_list_size = train_df.groupby('qid').size().max()
        self.train_frac = train_frac
        self.frac_queries = int(len(self.uniq_qids) * train_frac)
        # randomly select a fraction of queries for training logging policy
        np.random.seed(42)
        train_qids = np.random.choice(self.uniq_qids, self.frac_queries, replace=False)
        # randomly select n_train_queries number of queries for training logging policy
        # np.random.seed(42)
        # train_qids = np.random.choice(self.uniq_qids, n_train_queries, replace=False)
        # filter out queries not selected for training
        #self.remaining_queries = np.array(list(set(self.uniq_qids).difference(self.train_qids)))
        # overwrite training query set
        self.uniq_qids = train_qids
        self.df = train_df
        self.k = k

    def __len__(self):
        return len(self.uniq_qids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        qid_batch = self.uniq_qids[idx]
        qid_df = self.df[self.df.qid.isin([qid_batch])].groupby('qid').agg(lambda x: list(x))
        #qid_df['dcg_norm'] = qid_df.apply(lambda x:self.__dcg_norm(x['label']),axis=1)
        qid_df['dcg_norm'] = qid_df['label'].apply(self.__dcg_norm, args=(self.k,))
        #print(qid_df['dcg_norm'])
        qid_labels = np.array(qid_df['label'].tolist())
        qid_feats = np.array(qid_df['feats'].tolist())
        qid_norm = np.array(qid_df['dcg_norm'].tolist())
        # padding document labels to match max_list_size. TO-DO: Check what others do? 
        # find the gap between current query size and max_list_size
        #print(qid_labels.shape)
        gap_doc_size = self._max_list_size - qid_labels.shape[1]
        #print(gap_doc_size)
        # if no gap, then return
        if gap_doc_size == 0:
            qid_mask = np.full(qid_labels.shape[1], True)
            sample = {'labels': qid_labels, 'feats': qid_feats, 'mask':qid_mask, 'dcg_norm':qid_norm, 'qid': qid_batch}
            return sample
        else:
            qid_labels = np.hstack((qid_labels, np.zeros((1, gap_doc_size))))
            qid_feats = np.hstack((qid_feats, np.zeros((1, gap_doc_size, qid_feats.shape[2]))))
            qid_mask = np.full(qid_labels.shape[1], False)
            qid_mask[:-gap_doc_size] = True
            sample = {'labels': qid_labels, 'feats': qid_feats, 'mask': qid_mask, 'dcg_norm':qid_norm, 'qid': qid_batch}
            return sample


    
if __name__ == '__main__':
    
    ltr_dataloader = LTRLoggerDataLoader(qrel_path='<path_to/LTR_datasets/Yahoo/Fold1/train.txt', k=5, max_cand_size=120)
    train_dataloader = DataLoader(ltr_dataloader, batch_size=8,
                        shuffle=True, num_workers=4)
    
    for i_batch, sample in enumerate(train_dataloader):
        print(sample)

