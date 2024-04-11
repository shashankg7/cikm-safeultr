# -*- coding: utf-8 -*-
from sklearn import datasets
import pandas as pd
import numpy as np

import torch, os
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, FunctionTransformer


class LTRDataLoader(Dataset):
    '''
    Custom dataloader for the LTR datasets. 
    Reads QRELs in SVMLIGHT format and returns a dataloader for the same.
    '''
    @staticmethod
    def __num_labels_per_query(label_vector):
        return sum(label_vector)
    
    @staticmethod
    def __log_transform(x):
        return np.log(1 + np.abs(x)) * np.sign(x)

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

    def __init__(self, qrel_path, k, max_cand_size, noise, train=True, scaler=None, normalize=False, meta_dir=None, mode='val', save=False):
        '''
        Load the qrel file from the corresponding folder specified by the user
        max_cand_size: max. number of candidates per query
        scaler: sklearn's feat. normalizer
        meta_dir: directory to store the meta query df with qid, rel_labels, and feats. 
        mode: val/test - for saving val/test meta-df.
        '''
        self.qrel_path = qrel_path
        self.k = k
        # max number of candidates, (pre) computed from the training data
        libsvm_data = datasets.load_svmlight_file(self.qrel_path, query_id=True)
        X, y, qid = libsvm_data[0], libsvm_data[1], libsvm_data[2] 
        self.nqueries = np.unique(qid).shape[0]
        self.df = pd.DataFrame({'label' : y.T,
                                 'qid' : qid.T }, 
                                  columns=['label', 'qid'])
        # z-score normalize the feat vectors. TO-DO: DO query-wise normalization later
        #X = X.todense()
        X = X.toarray()
        if train == True and normalize==True:
            #self.scaler = StandardScaler(with_mean=False)
            #self.scaler = MinMaxScaler()
            self.scaler = FunctionTransformer(self.__log_transform)
            #self.scaler.fit(X)
            X = self.scaler.transform(X)
        elif train == False and normalize == True:
            X = scaler.transform(X)
        else:
            pass
        self.df['feats'] = X.tolist()
        # truncating per query doc size to a fixed size of 'k'. 
        # Currently randomly sampling k elements per query, have to look for a better alternative next time
        self.df = self.df.groupby('qid', group_keys=False)\
                           .apply(lambda x: x.sample(max_cand_size, replace=False, random_state=42) if len(x) > max_cand_size else x)
        # Binarize the labels (Joachims's WSDM'17 approach. )
        # uniq_labels = self.df['label'].unique()
        # label_cutoff = sorted(uniq_labels)[-2:][0]
        # self.df.loc[self.df['label'] <label_cutoff, 'label'] = 0
        # self.df.loc[self.df['label'] >=label_cutoff, 'label'] = 1
        # self.df.loc[self.df['label'] >=1, 'label'] = 1

        # Binarize the labels (WSDM'21 Unifying Online and offline paper's approach. Convert labels into P(R=1|q,d) )
        # if mode == 'train' or mode =='val':
        #     rel_scale = (1. - noise)/40.
        # else:
        #     rel_scale = 0.25
        # #rel_scale = 0.025
        # self.df['label'] = self.df['label'] * rel_scale
        train_queries = self.df.groupby('qid').agg(lambda x: list(x)).reset_index()
        # Filter out query which have no relevance labels
        train_queries['num_labels'] = train_queries.apply(lambda x:\
                                                          self.__num_labels_per_query(x['label']),\
                                                          axis=1)
        # save query meta df
        self.train_queries = train_queries
        meta_df_file = os.path.join(meta_dir, mode + '.pickle')
        
        qid_rel = train_queries[train_queries['num_labels'] >= 0.].qid.tolist()
        self.df = self.df[self.df.qid.isin(qid_rel)]
        train_queries = self.df.groupby('qid').agg(lambda x: list(x)).reset_index()
        if (not os.path.isfile(meta_df_file)) and (save):
            train_queries.to_pickle(meta_df_file)
        self.uniq_qids = np.unique(self.df['qid'].tolist())
        self.qid_map = dict({k:v for k, v in zip(sorted(self.uniq_qids), list(range(len(self.uniq_qids))))})
        self._max_list_size = self.df.groupby('qid').size().max()

    
    def set_label(self, rel_scale):
        '''
        change df['label] based on rel_scale
        '''
        self.df['label'] = self.df['label'] * rel_scale

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

