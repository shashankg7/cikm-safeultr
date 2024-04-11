
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

class PLRanker(pl.LightningModule):
    def __init__(self, num_samples, k, lr, optimizer, alpha, beta, out_dir, out_dir_val, meta_dir, train_qid_map, num_docs, policy_file, **MLP_args):
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
        self.train_qid_map = train_qid_map
        self.num_docs = num_docs
        self.policy_file = policy_file
        
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
        # expand rel_labels to match sampled ranking scores tensor size
        label_size = rel_labels.shape
        rel_labels = rel_labels.expand(label_size[0], self.num_samples, label_size[2])
        rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
        doc_scores = doc_scores.unsqueeze(1).expand(size[0], self.num_samples, size[1])
        doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
        log_scores, _   = self.pl_sampler.log_scores(doc_scores, mask, k=self.k)
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(self.device)
        # LTR metric/objective func - DCG with current weights_per_rank
        # TO-DO: Make it configurable, user should be able to specify the metric
        obj = rel_labels * weights_per_rank
        # take top-k elements
        #pdb.set_trace()
        obj = obj[:, :, :self.k].sum(-1)
        cv = obj.mean(-1).reshape(-1, 1)
        # weigh by the log-scores - REINFORCE trick
        obj = log_scores * (obj/dcg_norm - cv)
        # aggregate scores
        obj = -torch.mean(torch.sum(obj, dim=-1))
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
        test_ndcg = obj.detach().to('cpu').numpy()
        self.policy_file.write(str(test_ndcg))
        self.log("test-dcg@5", obj)



class PLRankerRisk(pl.LightningModule):
    def __init__(self, num_queries, num_samples, k, lr, optimizer, alpha, beta, out_dir, out_dir_val, meta_dir, train_qid_map, num_docs, policy_file, **MLP_args):
        super().__init__()
        #self.save_hyperparameters()
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
        #self.weights_per_rank = 1/np.log2(np.arange(k)+2)
        self.weights_per_rank = 1/(np.arange(k)+1)
        # flag to tell weather click simulation is for training data/validation data
        self.click_sim_val = False
        self.reg_weight = 1.
        self.train_qid_map = train_qid_map
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.policy_file = policy_file
        
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
        #weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = (1/(torch.arange(label_size[2])+1))
        weights_per_rank = weights_per_rank.to(self.device)
        alpha = batch['alpha']
        batch_size = alpha.shape[0]
        prop_counter = torch.zeros(batch_size, label_size[2]).to(self.device)
        prop_counter.requires_grad = False
        ix = torch.repeat_interleave(torch.arange(batch_size), self.num_samples)
        for i in range(self.k):
            iy = sampled_rankings[:, :, i].reshape(batch_size * self.num_samples)
            prop_counter.index_put_((ix, iy), torch.tensor(1/torch.log2(torch.tensor(i+2))), accumulate=True)
        prop_counter /= self.num_samples
        reg_denom = torch.sqrt(torch.mean((torch.square(prop_counter/alpha) * alpha).sum(-1)))
        alpha = alpha.unsqueeze(1).expand(batch_size, self.num_samples, -1)
        alpha = torch.gather(alpha, 2, sampled_rankings)
        prop_counter = prop_counter.unsqueeze(1).expand(label_size[0], self.num_samples, label_size[2])
        prop_counter = torch.gather(prop_counter, 2, sampled_rankings)
        # LTR metric/objective func - DCG with current weights_per_rank
        # TO-DO: Make it configurable, user should be able to specify the metric
        ips_weight = prop_counter/alpha
        ips_weight.nan_to_num_()
        weights_per_rank[self.k:] = 0.
        #ips_weight = ips_weight/ips_weight.sum(-1).unsqueeze(-1)
        grad_weight = (ips_weight * weights_per_rank).sum(-1)
        #grad_weight = (ips_weight * weights_per_rank)
        #out, _ = torch.max(grad_weight, dim=-1)
        #grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
        #grad_weight = grad_weight * log_scores
        #with torch.no_grad():
        #grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
        #grad_weight = grad_weight/out.reshape(-1, 1)
        #reg = torch.mean(((ips_weight * weights_per_rank).sum(-1) * log_scores).sum(-1))
        #target = torch.ones_like(doc_scores)
        #reg = -torch.mean((grad_weight * log_scores).sum(-1)) #- 0.01 * entropy_loss(doc_scores, target)
        #reg = -torch.mean(grad_weight.sum(-1)) #- 0.01 * entropy_loss(doc_scores, target)
        #obj = rel_labels * weights_per_rank
        # take top-k elements
        #pdb.set_trace()
        #obj = obj[:, :, :self.k].sum(-1)
        # Control variate for variance.
        cv = grad_weight.mean(-1).reshape(-1, 1)     
        #cv1 = obj.mean(-1).reshape(-1, 1)   
        #obj = log_scores * ((obj - 10/np.sqrt(self.num_queries) * grad_weight).sum(-1))
        # weigh by the log-scores - REINFORCE trick
        #obj = log_scores * (obj -  0.025 * (grad_weight - cv))
        obj = log_scores * ( grad_weight - cv) 
        #obj = log_scores * ( grad_weight ) #- 0.05 * entropy_reg
        #obj = log_scores * (obj/dcg_norm - 2/(np.sqrt(self.num_queries)) *  grad_weight)
        #obj = log_scores * (obj/dcg_norm - 1/(np.sqrt(self.num_queries)* reg_denom) * grad_weight)
        # aggregate scores
        obj = torch.mean(torch.sum(obj, dim=-1)) #+ 1e-2 * entropy_reg
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
        test_ndcg = obj.detach().to('cpu').numpy()
        self.policy_file.write(str(test_ndcg))
        self.log("test-dcg@5", obj)


        
        

