# -*- coding: utf-8 -*-
from sklearn import datasets
import pandas as pd
import numpy as np

import torch, os
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LTRClickDataLoader(Dataset):
    '''
    Child class of the full-information dataloader. 
    Used for click simulation
    '''
    def __dcg_norm(self, label_vector, k):
        '''
        get the dcg normalizing factor to calculate ndcg@k
        '''
        label_vector = np.array(label_vector)
        label_vector = sorted(label_vector, reverse=True)
        weights_per_rank = 1/np.log2(np.array(range(len(label_vector))) +2)
        label_vector *= weights_per_rank
        return np.sum(label_vector[:k])
    
    def __init__(self, num_sessions, train_df, k):
        self.uniq_qids = np.unique(train_df['qid'].tolist())
        self.qid_map = dict({k:v for k, v in zip(sorted(self.uniq_qids), list(range(len(self.uniq_qids))))})
        self._max_list_size = train_df.groupby('qid').size().max()
        # random sample (with replacement) num_sessions queries from the simulation queries
        self.sampled_queries = np.random.choice(self.uniq_qids, num_sessions)
        sampled_qids, counts = np.unique(self.sampled_queries, return_counts=True)
        self.uniq_qids = sampled_qids
        self.qid_counts = counts
        self.k = k
        self.df = train_df

    @property            
    def max_list_size(self): 
        '''
        getter function to return max_list_size
        '''
        return self._max_list_size
        
    @max_list_size.setter
    def max_list_size(self, value):
        '''
        Set Max list size from outside of the class. 
        max_list_size = max(train_list_size, val_list_size, test_list_size)
        '''
        self._max_list_size = value

    def __len__(self):
        return len(self.uniq_qids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        qid_batch = self.uniq_qids[idx]
        qid_freq = self.qid_counts[idx]
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
            sample = {'labels': qid_labels, 'feats': qid_feats, 'mask':qid_mask, 'dcg_norm':qid_norm, 'qid': qid_batch, 'qid_freq':qid_freq}
            return sample
        else:
            qid_labels = np.hstack((qid_labels, np.zeros((1, gap_doc_size))))
            qid_feats = np.hstack((qid_feats, np.zeros((1, gap_doc_size, qid_feats.shape[2]))))
            qid_mask = np.full(qid_labels.shape[1], False)
            qid_mask[:-gap_doc_size] = True
            sample = {'labels': qid_labels, 'feats': qid_feats, 'mask': qid_mask, 'dcg_norm':qid_norm, 'qid': qid_batch, 'qid_freq':qid_freq}
            return sample




    
if __name__ == '__main__':
    
    ltr_dataloader = LTRLoggerDataLoader(qrel_path='<path_to/LTR_datasets/Yahoo/Fold1/train.txt', k=5, max_cand_size=120)
    train_dataloader = DataLoader(ltr_dataloader, batch_size=8,
                        shuffle=True, num_workers=4)
    
    for i_batch, sample in enumerate(train_dataloader):
        print(sample)

