
from asyncio import base_tasks
from cmath import exp
import os, json, pdb
import pandas as pd
import numpy as np
from typing import Any, List, Optional


import torch.optim as optim
import torch
import pytorch_lightning as pl
import torch.nn as nn
from src.models.nnmodel import DocScorer
from src.utils.PlackettLuce import PlackettLuceModel
from pytorch_lightning.callbacks import BasePredictionWriter


class PLRankerAction(pl.LightningModule):
    def __init__(self, num_queries, num_samples, k, lr, optimizer, alpha, beta, out_dir, out_dir_val, meta_dir, train_qid_map, num_docs, clip=1e-4, **MLP_args):
        super().__init__()
        self.save_hyperparameters()
        self.num_samples = num_samples
        #self.doc_scorer = DocScorer(**MLP_args)
        self.doc_scorer = DocScorer(**MLP_args)
        self.k = k
        self.lr = lr
        self.optimizer = optimizer
        self.beta_tensor = torch.tensor(beta)#.to(self.device)
        self.alpha_tensor = torch.tensor(alpha)#.to(self.device)
        self.output_dir = out_dir
        self.output_dir_val = out_dir_val
        self.meta_dir = meta_dir
        self.pl_sampler = PlackettLuceModel(num_samples)
        # weights per rank, for computing the LTR metric
        self.weights_per_rank = 1/np.log2(np.arange(k)+2)
        # flag to tell weather click simulation is for training data/validation data
        self.click_sim_val = False
        self.reg_weight = 1.
        self.train_qid_map = train_qid_map
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.clip = torch.tensor(clip).to(self.device)
        
    def forward(self, x):
        scores = self.doc_scorer(x)
        return scores
    
    def configure_optimizers(self):
        if self.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'Adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.lr)
        elif self.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=self.lr)
        else:
            # if no option provided in config, use SGD
            optimizer = optim.SGD(self.parameters(), lr=0.0001)
        #optimizer = optim.Adadelta(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)  # Exponential decay over epochs
        return [optimizer], [scheduler]
    
    
    def training_step(self, batch, batch_idx):
        rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
        doc_feats = doc_feats.squeeze(1).float()
        doc_scores = self.doc_scorer(doc_feats).squeeze(-1)
        # generate samples from PL-model
        ranking_scores, sampled_rankings = self.pl_sampler.sample(doc_scores, mask)
        size = doc_scores.shape
        prob = nn.functional.softmax(doc_scores, dim=-1)
        log_scores = nn.functional.log_softmax(doc_scores, dim=-1)
        entropy_reg = -(prob * log_scores).sum(-1).mean()
        # expand rel_labels to match sampled ranking scores tensor size
        label_size = rel_labels.shape
        rel_labels = rel_labels.expand(label_size[0], self.num_samples, label_size[2])
        rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
        doc_scores = doc_scores.unsqueeze(1).expand(size[0], self.num_samples, size[1])
        doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
        log_scores, _   = self.pl_sampler.log_scores(doc_scores, mask, k=self.k)
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(self.device)
        doc_prob_rank = batch['alpha']
        batch_size = doc_prob_rank.shape[0]
        prop_counter = torch.zeros(batch_size, label_size[2]).to(self.device)
        prop_counter.requires_grad = False
        logging_policy = torch.ones(batch_size, self.num_samples).to(self.device)
        logging_policy.requires_grad = False
        ix = torch.repeat_interleave(torch.arange(batch_size), self.num_samples)
        for i in range(self.k):
            iy = sampled_rankings[:, :, i].reshape(batch_size * self.num_samples)
            rank_prob = torch.gather(input=doc_prob_rank[...,i].reshape(batch_size * label_size[2]), dim=0, index=iy).reshape(batch_size, self.num_samples)
            logging_policy *= rank_prob

        weights_per_rank[self.k:] = 0.
        logging_policy = torch.maximum(logging_policy, self.clip)
        grad_weight = (rel_labels * weights_per_rank).sum(-1)
        grad_weight = grad_weight/logging_policy
        #grad_weight = (ips_weight * weights_per_rank)
        #out, _ = torch.max(grad_weight, dim=-1)
        grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
        grad_weight.nan_to_num_()
        cv = grad_weight.mean(-1).reshape(-1, 1)     
        obj = log_scores * ( grad_weight - cv) 
        obj = -torch.mean(torch.sum(obj, dim=-1)) #+ 1e-2 * entropy_reg
        #obj += self.reg_weight * reg
        self.log("loss", obj, on_step=True, on_epoch=True)
        return obj
        
    def validation_step(self, batch, batch_idx):
        rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
        doc_feats = doc_feats.squeeze(1).float()
        doc_scores = self.doc_scorer(doc_feats).squeeze(-1)
        # generate samples from PL-model
        ranking_scores, sampled_rankings = self.pl_sampler.sample(doc_scores, mask)
        # expand rel_labels to match sampled ranking scores tensor size
        label_size = rel_labels.shape
        rel_labels = rel_labels.expand(label_size[0], self.num_samples, label_size[2])
        rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
        #log_scores = self.pl_sampler.log_scores(ranking_scores)
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(self.device)
        # LTR metric/objective func
        obj = rel_labels * weights_per_rank
        #pdb.set_trace()
        # take top-k elements
        obj = obj[:, :, :self.k].sum(-1)
        #pdb.set_trace()
        # aggregate scores
        obj = torch.mean(obj/dcg_norm, dim=-1)
        return obj
        
    def validation_epoch_end(self, outputs):
        #pdb.set_trace()
        obj = torch.hstack(outputs).mean()
        self.log("val-dcg@5", obj, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
        doc_feats = doc_feats.squeeze(1).float()
        doc_scores = self.doc_scorer(doc_feats).squeeze(-1)
        # generate samples from PL-model
        ranking_scores, sampled_rankings = self.pl_sampler.sample(doc_scores, mask)
        # expand rel_labels to match sampled ranking scores tensor size
        label_size = rel_labels.shape
        rel_labels = rel_labels.expand(label_size[0], self.num_samples, label_size[2])
        rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
        #log_scores = self.pl_sampler.log_scores(ranking_scores)
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(self.device)
        # LTR metric/objective func
        obj = rel_labels * weights_per_rank
        # take top-k elements
        obj = obj[:, :, :self.k].sum(-1)
        # aggregate scores
        obj = torch.mean(obj/dcg_norm, dim=-1)
        return obj
        
    def test_epoch_end(self, outputs):
        obj = torch.hstack(outputs).mean()
        self.log("test-dcg@5", obj)


class PLRankerRiskAction(pl.LightningModule):
    def __init__(self, num_queries, num_samples, k, lr, optimizer, alpha, beta, out_dir, out_dir_val, meta_dir, train_qid_map, num_docs, clip=1e-4, **MLP_args):
        super().__init__()
        self.save_hyperparameters()
        self.num_samples = num_samples
        #self.doc_scorer = DocScorer(**MLP_args)
        self.doc_scorer = DocScorer(**MLP_args)
        self.k = k
        self.lr = lr
        self.optimizer = optimizer
        self.beta_tensor = torch.tensor(beta)#.to(self.device)
        self.alpha_tensor = torch.tensor(alpha)#.to(self.device)
        self.output_dir = out_dir
        self.output_dir_val = out_dir_val
        self.meta_dir = meta_dir
        self.pl_sampler = PlackettLuceModel(num_samples)
        # weights per rank, for computing the LTR metric
        self.weights_per_rank = 1/np.log2(np.arange(k)+2)
        # flag to tell weather click simulation is for training data/validation data
        self.click_sim_val = False
        self.reg_weight = 1.
        self.train_qid_map = train_qid_map
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.clip = torch.tensor(clip).to(self.device)
        
    def forward(self, x):
        scores = self.doc_scorer(x)
        return scores
    
    def configure_optimizers(self):
        if self.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'Adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.lr)
        elif self.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=self.lr)
        else:
            # if no option provided in config, use SGD
            optimizer = optim.SGD(self.parameters(), lr=0.0001)
        #optimizer = optim.Adadelta(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)  # Exponential decay over epochs
        return [optimizer], [scheduler]
    
    
    def training_step(self, batch, batch_idx):
        rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
        doc_feats = doc_feats.squeeze(1).float()
        doc_scores = self.doc_scorer(doc_feats).squeeze(-1)
        # generate samples from PL-model
        ranking_scores, sampled_rankings = self.pl_sampler.sample(doc_scores, mask)
        size = doc_scores.shape
        prob = nn.functional.softmax(doc_scores, dim=-1)
        log_scores = nn.functional.log_softmax(doc_scores, dim=-1)
        entropy_reg = -(prob * log_scores).sum(-1).mean()
        # expand rel_labels to match sampled ranking scores tensor size
        label_size = rel_labels.shape
        rel_labels = rel_labels.expand(label_size[0], self.num_samples, label_size[2])
        rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
        doc_scores = doc_scores.unsqueeze(1).expand(size[0], self.num_samples, size[1])
        doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
        log_scores, _   = self.pl_sampler.log_scores(doc_scores, mask, k=self.k)
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(self.device)
        doc_prob_rank = batch['alpha']
        batch_size = doc_prob_rank.shape[0]
        prop_counter = torch.zeros(batch_size, label_size[2]).to(self.device)
        prop_counter.requires_grad = False
        logging_policy = torch.ones(batch_size, self.num_samples).to(self.device)
        logging_policy.requires_grad = False
        ix = torch.repeat_interleave(torch.arange(batch_size), self.num_samples)
        for i in range(self.k):
            iy = sampled_rankings[:, :, i].reshape(batch_size * self.num_samples)
            rank_prob = torch.gather(input=doc_prob_rank[...,i].reshape(batch_size * label_size[2]), dim=0, index=iy).reshape(batch_size, self.num_samples)
            logging_policy *= rank_prob

        logging_policy = torch.maximum(logging_policy, self.clip)
        grad_weight = log_scores/logging_policy
        #grad_weight = (ips_weight * weights_per_rank)
        #out, _ = torch.max(grad_weight, dim=-1)
        #grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
        cv = grad_weight.mean(-1).reshape(-1, 1)     
        obj = ( grad_weight - cv) 
        obj = obj.mean(-1).mean(-1)
        #obj += self.reg_weight * reg
        self.log("loss", obj, on_step=True, on_epoch=True)
        return obj
        
    def validation_step(self, batch, batch_idx):
        rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
        doc_feats = doc_feats.squeeze(1).float()
        doc_scores = self.doc_scorer(doc_feats).squeeze(-1)
        # generate samples from PL-model
        ranking_scores, sampled_rankings = self.pl_sampler.sample(doc_scores, mask)
        # expand rel_labels to match sampled ranking scores tensor size
        label_size = rel_labels.shape
        rel_labels = rel_labels.expand(label_size[0], self.num_samples, label_size[2])
        rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
        #log_scores = self.pl_sampler.log_scores(ranking_scores)
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(self.device)
        # LTR metric/objective func
        obj = rel_labels * weights_per_rank
        #pdb.set_trace()
        # take top-k elements
        obj = obj[:, :, :self.k].sum(-1)
        #pdb.set_trace()
        # aggregate scores
        obj = torch.mean(obj/dcg_norm, dim=-1)
        return obj
        
    def validation_epoch_end(self, outputs):
        #pdb.set_trace()
        obj = torch.hstack(outputs).mean()
        self.log("val-dcg@5", obj, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
        doc_feats = doc_feats.squeeze(1).float()
        doc_scores = self.doc_scorer(doc_feats).squeeze(-1)
        # generate samples from PL-model
        ranking_scores, sampled_rankings = self.pl_sampler.sample(doc_scores, mask)
        # expand rel_labels to match sampled ranking scores tensor size
        label_size = rel_labels.shape
        rel_labels = rel_labels.expand(label_size[0], self.num_samples, label_size[2])
        rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
        #log_scores = self.pl_sampler.log_scores(ranking_scores)
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(self.device)
        # LTR metric/objective func
        obj = rel_labels * weights_per_rank
        # take top-k elements
        obj = obj[:, :, :self.k].sum(-1)
        # aggregate scores
        obj = torch.mean(obj/dcg_norm, dim=-1)
        return obj
        
    def test_epoch_end(self, outputs):
        obj = torch.hstack(outputs).mean()
        self.log("test-dcg@5", obj)


        
        

