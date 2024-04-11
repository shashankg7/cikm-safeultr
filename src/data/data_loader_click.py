# -*- coding: utf-8 -*-
from sklearn import datasets
import pandas as pd
import numpy as np
import glob, copy
import pickle

import torch, os
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ClickLogDataloader(Dataset):
    '''
    Datalaoder for click logs. 
    '''
    @staticmethod
    def __num_labels_per_query(label_vector):
        return sum(label_vector)
    
    def __dcg_norm(self, click_vector):
        '''
        get the dcg normalizing factor to calculate ndcg@k
        '''
        label_vector = sorted(label_vector, reverse=True)
        weights_per_rank = 1/np.log2(np.array(range(len(label_vector))) +2)
        label_vector *= weights_per_rank
        return np.sum(label_vector[:k])

    def set_query_feat(self, qid, feat_vec):
        '''
        sets self.query_feat_vec with values from feat_vec
        '''
        feat_vec = np.array(feat_vec)
        dim = feat_vec.shape[1]
        gap = self.max_cand_size - feat_vec.shape[0]
        if gap == 0:
            qid_mask = np.full(feat_vec.shape[0], True)
            self.qid_feat_tensor[self.qid_map[qid], :,:] = feat_vec
            self.qid_mask[self.qid_map[qid], :] = qid_mask
        else:
            feat_vec = np.vstack((feat_vec, np.zeros((gap, dim))))
            qid_mask = np.full(feat_vec.shape[0], False)
            qid_mask[:-gap] = True
            self.qid_mask[self.qid_map[qid], :] = qid_mask
            self.qid_feat_tensor[self.qid_map[qid], :, :] = feat_vec
    
    def set_expect_exp(self, qid, exp, type='alpha'):
        '''
        sets exposure vector from dataframe to numpy matrix
        '''
        feat_vec = np.array(eval(exp))
        dim = feat_vec.shape[0]
        if type == 'alpha':
            self.expected_alpha[self.qid_map[qid], :] = feat_vec
        else:
            self.expected_beta[self.qid_map[qid], :] = feat_vec

    def __init__(self, meta_dir=None, click_dir=None, mode='train', feat_vec_dim=None, max_cand_size=None, clip=0.00001, clip_alpha=1e-9, click_model='pbm', estimator='ips'):
        '''
        Loads the click data, include clicks generated, sampled rankings, qids and the feats. 
        '''
        self.click_model = click_model
        self.click_dir = click_dir
        sampled_rankings = glob.glob(click_dir + '/ranking*')
        clicks = glob.glob(click_dir + '/click*')
        qids = glob.glob(click_dir + '/qids*')
        masks = glob.glob(click_dir + '/mask*')
        self.sampled_rankings = sampled_rankings
        self.clicks = clicks
        self.qids = qids
        self.clip = clip
        self.estimator = estimator
        train_qids = pd.read_pickle(os.path.join(meta_dir,  'train.pickle'))
        val_qids = pd.read_pickle(os.path.join(meta_dir,  'val.pickle'))
        #test_qids = pd.read_csv(os.path.join(meta_dir,  'test.csv'))
        self.max_cand_size = max_cand_size
        # global tensor with qid-doc feat vectors (for training qids)
        with open(os.path.join(meta_dir, 'train_qid_map'), 'rb') as fp:
            self.qid_map = pickle.load(fp)
        #doc_prob_per_rank = np.load(os.path.join(meta_dir, 'doc_prob_per_rank.npy'))
        self.qid_feat_tensor = np.zeros((len(self.qid_map), max_cand_size, feat_vec_dim))
        self.qid_mask = np.full((len(self.qid_map), max_cand_size), True)
        self.expected_alpha = np.load(os.path.join(meta_dir, 'alpha.npy'))
        self.expected_alpha1 = copy.deepcopy(self.expected_alpha)
        self.expected_beta = np.load(os.path.join(meta_dir, 'beta.npy'))
        self.expected_alpha_beta = np.load(os.path.join(meta_dir, 'alpha_beta.npy'))
        self.ctr_map = np.load(os.path.join(meta_dir, 'ctr_map.npy'))
        self.display_mask = np.load(os.path.join(meta_dir, 'display_mask.npy'))
        self.alpha_freq = np.load(os.path.join(meta_dir, 'alpha_freq.npy'))
        self.beta_freq = np.load(os.path.join(meta_dir, 'beta_freq.npy'))
        # load train/val active queries, depending on the 'mode' flag
        if mode == 'train':
            self.qid_active = np.load(os.path.join(meta_dir, 'query_train.npy'))
            #self.qid_active = np.where(self.query_freq > 0)[0]
        else:
            self.qid_active = np.load(os.path.join(meta_dir, 'query_val.npy'))
        self.expected_alpha = np.maximum(self.expected_alpha, self.clip)
        self.expected_alpha1 = np.maximum(self.expected_alpha1, clip_alpha)
        self.expected_alpha_beta = np.maximum(self.expected_alpha_beta, clip_alpha)
        #self.expected_beta = np.maximum(self.expected_beta, clip_alpha)
        #self.qid_map_rev = dict({v:k for k,v in self.qid_map.items()})
        #self.qid_active_idx = np.array(list(map(lambda x:self.qid_map[self.qid_map_rev[x]], self.qids_active)))
        if click_model == 'pbm':
            self.ips_weights = (self.ctr_map/self.expected_alpha)
        else:
            if self.estimator == 'ips':
                self.ips_weights = (self.ctr_map - self.expected_beta)/self.expected_alpha
                # self.ips_weights = self.ips_weights/self.ips_weights.sum(-1).reshape(-1, 1)
                np.nan_to_num(self.ips_weights, copy=False)
            else:
                self.ips_weights = (self.ctr_map - self.beta_freq)/self.expected_alpha
            self.reg_weights = (self.ctr_map - self.beta_freq)/self.expected_alpha
            #self.ips_weights = self.ctr_map 
        np.nan_to_num(self.ips_weights, copy=False)
        train_qids.apply(lambda x: self.set_query_feat(x.qid, x.feats), axis=1)
        self.qid_feat_tensor = self.qid_feat_tensor[self.qid_active, :, :]
        self.ips_weights = self.ips_weights[self.qid_active, :]
        self.reg_weights = self.reg_weights[self.qid_active, :]
        self.qid_mask = self.qid_mask[self.qid_active, :]
        self.expected_alpha1 = self.expected_alpha1[self.qid_active, :]
        self.expected_alpha = self.expected_alpha[self.qid_active, :]
        self.expected_beta = self.expected_beta[self.qid_active, :]
        self.expected_alpha_beta = self.expected_alpha_beta[self.qid_active, :]
        self.display_mask = self.display_mask[self.qid_active, :]
        self.alpha_freq = self.alpha_freq[self.qid_active, :]
        self.beta_freq = self.beta_freq[self.qid_active, :]
        #self.expected_alpha = self.expected_alpha/self.expected_alpha.sum(-1).reshape(-1, 1)
        #np.nan_to_num(self.expected_alpha, copy=False)

    
    def __len__(self):
        return self.qid_feat_tensor.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        qid_labels = self.ips_weights[idx, :].reshape(1, -1)
        qid_labels_reg = self.reg_weights[idx, :].reshape(1, -1)
        qid_mask = self.qid_mask[idx, :]
        qid_feats = np.expand_dims(self.qid_feat_tensor[idx, :, :], 0)
        qid_batch = np.zeros_like(idx)
        qid_norm = np.ones(1)
        #qid_alpha = self.expected_alpha1[idx, :]
        qid_alpha = self.expected_alpha_beta[idx, :]
        # get rho(d) for DR estimator
        qid_rho = self.expected_alpha[idx, :]
        display_mask = self.display_mask[idx, :]
        alpha_freq = self.alpha_freq[idx, :]
        beta_freq = self.beta_freq[idx, :]
        qid_rho1 = self.expected_beta[idx, :]
        sample = {'labels': qid_labels, 'labels_reg':qid_labels_reg, 'feats': qid_feats, 'mask':qid_mask, 'dcg_norm':qid_norm,\
                    'qid': qid_batch, 'alpha':qid_alpha, 'display_mask': display_mask,\
                        'qid_rho':qid_rho, 'alpha_freq':alpha_freq, 'beta_freq':beta_freq, \
                            'qid_rho1': qid_rho1}
        return sample
        



    
if __name__ == '__main__':
    
    ltr_dataloader = LTRLoggerDataLoader(qrel_path='<path_to/LTR_datasets/Yahoo/Fold1/train.txt', k=5, max_cand_size=120)
    train_dataloader = DataLoader(ltr_dataloader, batch_size=8,
                        shuffle=True, num_workers=4)
    
    for i_batch, sample in enumerate(train_dataloader):
        print(sample)

