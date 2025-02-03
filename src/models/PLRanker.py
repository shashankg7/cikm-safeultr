
from cmath import exp
import os, json, pdb, gc
import pandas as pd
import numpy as np
from typing import Any, List, Optional


import torch.optim as optim
import torch
import pytorch_lightning as pl
import torch.nn as nn
from src.models.nnmodel import DocScorer, Logger
from src.utils.PlackettLuce import PlackettLuceModel
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.distributions.binomial import Binomial

class PLRanker(pl.LightningModule):
    def __init__(self, num_samples, k, lr, optimizer, alpha, alpha_exp, beta, out_dir, out_dir_val, meta_dir, train_qid_map, num_docs, noise, click_model_type,  deterministic=False, eps_greedy=0.5, T=1, noise_click=0.3, **MLP_args):
        super().__init__()
        self.save_hyperparameters()
        self.num_samples = num_samples
        #self.doc_scorer = Logger(**MLP_args)
        self.doc_scorer = DocScorer(**MLP_args)
        self.k = k
        self.lr = lr
        self.optimizer = optimizer
        self.beta_tensor = torch.tensor(beta)#.to(self.device)
        self.alpha_tensor = torch.tensor(alpha)#.to(self.device)
        self.alpha_tensor_exp = torch.tensor(alpha_exp)
        self.output_dir = out_dir
        self.output_dir_val = out_dir_val
        self.meta_dir = meta_dir
        self.pl_sampler = PlackettLuceModel(num_samples)
        # weights per rank, for computing the LTR metric
        self.weights_per_rank = 1/np.log2(np.arange(k)+2)
        # flag to tell weather click simulation is for training data/validation data
        self.click_sim_val = False
        self.train_qid_map = train_qid_map
        self.doc_per_rank_prob = torch.zeros((len(train_qid_map), num_docs, k))
        self.doc_per_rank_prob_det = torch.zeros((len(train_qid_map), num_docs, k))
        self.doc_per_rank_prob_beta = torch.zeros((len(train_qid_map), num_docs, k))
        self.alpha_freq = torch.zeros((len(train_qid_map), num_docs))
        self.beta_freq = torch.zeros((len(train_qid_map), num_docs))
        self.qid_clicks = torch.zeros((len(train_qid_map), num_docs, k))
        self.qid_clicks = self.qid_clicks.to(self.device)
        self.query_freq = torch.zeros((len(train_qid_map)))
        self.num_docs = num_docs
        self.noise = noise
        self.eps = 1e-08
        self.T = T
        self.deterministic = deterministic
        self.eps_greedy = eps_greedy
        self.alpha = torch.tensor(alpha).to(self.device)
        self.click_model_type = click_model_type
        self.noise_click = noise_click
        
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
        #weights_per_rank = (1/(torch.arange(label_size[2])+1))
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(self.device)
        weights_per_rank[self.k:] =0.
        # LTR metric/objective func - DCG with current weights_per_rank
        # TO-DO: Make it configurable, user should be able to specify the metric
        obj = rel_labels * weights_per_rank
        # take top-k elements
        #pdb.set_trace()
        obj = obj[:, :, :self.k].sum(-1)
        obj/= dcg_norm
        cv = obj.mean(-1).reshape(-1, 1)
        # weigh by the log-scores - REINFORCE trick
        #obj = log_scores * (obj/dcg_norm)
        obj = log_scores * (obj- cv)
        # aggregate scores
        obj = -torch.mean(torch.mean(obj, dim=-1))
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
        #weights_per_rank = (1/(torch.arange(label_size[2])+1))
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(self.device)
        weights_per_rank[self.k:] =0.
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
        weights_per_rank[self.k:] =0.
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

    
    @staticmethod
    def get_expected_exp(sampled_rankings, doc_prob_per_rank):
        '''
        get expected exposure for all documents in the sampled rankings. 
        output size: #queries * n_docs (cutoff)
        '''
        n_queries = sampled_rankings.shape[0]
        for query in range(n_queries):
            query_rankings = 0


    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        '''
        Takes the output from the 'predict_step' function, generate clicks and dump the (clicks, feats, ranking) tuple on disk. 
        TO-DO: Dumping feats. on disk is not efficient, and redundant. 
               Next step is to generate just the clicks, rankings, doc_ids and join with doc feats on the fly. 
        '''
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        '''
        Prediction step to get sampled rankings for a given batch. 
        Used in click simulation step.
        '''
        #self.qid_clicks = self.qid_clicks.to(self.device)
        rel_labels, doc_feats, mask, dcg_norm, qids = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm'], batch['qid']
        qid_freq = batch['qid_freq']
        qids = qids.detach().to('cpu').numpy()
        qids_mapped = torch.tensor([self.train_qid_map[x] for x in qids]).to(self.device)
        doc_feats = doc_feats.squeeze(1).float()
        doc_scores = self.doc_scorer(doc_feats).squeeze(-1)
        self.query_freq.index_put_((qids_mapped,), qid_freq.float().to('cpu'), accumulate=True)
        for i, qid in enumerate(qids):
            query_score = doc_scores[i, :]/self.T
            if not self.deterministic:
                # when the logging policy is stochastic
                query_score = query_score.repeat(qid_freq[i], 1)
                policy_score = query_score.clone().detach() 
                mask_qid = mask[i, :]
                mask_qid = mask_qid.repeat(qid_freq[i], 1)
                rel_labels_qid = rel_labels[i, :].squeeze(0)
                rel_labels_qid = rel_labels_qid.repeat(qid_freq[i], 1)
                unif = torch.rand_like(query_score)
                gumbel_scores = query_score - torch.log(-torch.log(unif + self.eps ) )
                gumbel_scores = gumbel_scores * mask_qid
                _, sampled_rankings = torch.sort(gumbel_scores, descending=True, dim=-1, stable=True)
                policy_score[torch.arange(qid_freq[i]).view(-1,1), sampled_rankings[:, :(self.k-1)]] = torch.tensor(-torch.inf)
                rel_labels_qid = torch.gather(rel_labels_qid, 1, sampled_rankings)
                noise = torch.zeros_like(rel_labels_qid)
                noise[:, :self.k] = self.noise
                if self.click_model_type == 'pbm':
                    click_prob = self.alpha_tensor.to(self.device) * rel_labels_qid + noise
                else:
                    click_prob = self.alpha_tensor.to(self.device) * rel_labels_qid + self.beta_tensor.to(self.device) 
                    #noise_click = torch.zeros_like(click_prob)
                    #noise_click[:, :self.k] = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.1]).to(self.device)
                    #click_prob += noise_click.to(self.device)
                #click_prob =  self.noise #+ self.beta_tensor.to(self.device)
                # generate clicks with noise
                #unif_noise = torch.rand_like(click_prob) 
                #noise_mask = (unif_noise <= (1 - self.noise_click)).to(torch.int)
                # Controlled adversarial setup.
                # adversarial mask 
                adv_mask = torch.zeros_like(click_prob).bernoulli_(0.0)
                # with prob. eps, adversarial clicks are activated, and with prob. (1-eps), normal clicks. 
                # The higher the eps, the higher the noise in clicks.
                # eps=1 -> pure adversarial setup, eps=0 -> normal setup
                # higher the eps, higher the noise. 
                click_prob = adv_mask * (1-click_prob) + (1- adv_mask) * click_prob
                clicks = torch.bernoulli(click_prob)
                #clicks = torch.bernoulli(1 - click_prob) #* noise_mask
                #clicks = torch.ones_like(click_prob) * (1- noise_mask)
                # USING actual click probl
                #clicks = click_prob
                ix = torch.repeat_interleave(qids_mapped[i], qid_freq[i])
                for k in range(self.k):
                    iy = sampled_rankings[:, k]
                    # instead of recording the propensity/impression, record the alpha[k], beta[k] values
                    #self.doc_per_rank_prob_det[:, :, k].index_put_((ix, iy), torch.tensor(1.), accumulate=True)
                    self.doc_per_rank_prob_det[:, :, k].index_put_((ix, iy), self.alpha_tensor[k].float(), accumulate=True)
                    self.doc_per_rank_prob_beta[:, :, k].index_put_((ix, iy), self.beta_tensor[k].float(), accumulate=True)
                    clicks_k = clicks[:, k].float().to('cpu')
                    self.qid_clicks[:, :, k].index_put_((ix, iy), clicks_k, accumulate=True)
                
                for k in range(self.k-1):
                    iy = sampled_rankings[:, k]
                    self.doc_per_rank_prob[:, :, k].index_put_((ix, iy), torch.tensor(1.), accumulate=True)
            
                self.doc_per_rank_prob[qids_mapped[i], :, :(self.k-1)]/=qid_freq[i].to('cpu')   
                self.doc_per_rank_prob[qids_mapped[i], :, self.k-1] = torch.mean(torch.softmax(policy_score, 1), 0)
                
            else:
                # when the logging policy is deterministic
                query_score = query_score.repeat(qid_freq[i], 1)
                mask_qid = mask[i, :]
                mask_qid = mask_qid.repeat(qid_freq[i], 1)
                rel_labels_qid = rel_labels[i, :].squeeze(0)
                rel_labels_qid = rel_labels_qid.repeat(qid_freq[i], 1)
                gumbel_scores = query_score 
                gumbel_scores = gumbel_scores * mask_qid
                _, sampled_rankings = torch.sort(gumbel_scores, descending=True, dim=-1, stable=True)
                # first k-1 elements selected deterministically
                click_prob = self.alpha_tensor_exp.to(self.device)[:(self.k-1)] * rel_labels_qid[:, :(self.k-1)]  #+ self.beta_tensor.to(self.device)
                clicks = torch.bernoulli(click_prob)
                ix = torch.repeat_interleave(qids_mapped[i], qid_freq[i])
                for k in range(self.k-1):
                    iy = sampled_rankings[:, k]
                    self.doc_per_rank_prob_det.index_put_((ix, iy), self.alpha_tensor_exp[k].to(torch.float), accumulate=False)
                    clicks_k = clicks[:, k].float().to('cpu')
                    self.qid_clicks[:, :, k].index_put_((ix, iy), clicks_k, accumulate=True)
                num_docs = gumbel_scores.shape[1]
                # pick item for the last position with eps-greedy stratergy. 
                eps_greedy = np.random.random(qid_freq[i])
                eps_greedy_flag = eps_greedy < self.eps_greedy
                #if eps_greedy < self.eps:
                # select the Kth position randomly, keeping in account the mask length. 
                #kth_sample_index = (torch.randint(0, num_docs - (self.k - 1), (qid_freq[i],) ).to(self.device)) * eps_greedy_flag + self.k - 1
                high = mask_qid.sum(-1).to('cpu').numpy()
                num_docs = high[0]
                kth_sample_index = torch.tensor(np.random.randint(0, high - self.k + 1)) * eps_greedy_flag + self.k - 1
                kth_sample_index = kth_sample_index.to(self.device)
                truncated_rankings = sampled_rankings[:, self.k:]
                kth_sample = torch.gather(sampled_rankings, 1, kth_sample_index.unsqueeze(0))
                self.doc_per_rank_prob_det[qids_mapped[i], sampled_rankings[:, self.k-1]] = (self.alpha_tensor_exp[self.k-1] * (self.eps_greedy * (1/(num_docs - self.k + 1)) + (1 - self.eps_greedy))).to(torch.float)
                #self.doc_per_rank_prob_det[qids_mapped[i], kth_sample] = qid_freq[i] * (self.alpha_tensor[self.k-1] * (self.eps_greedy * (1/(num_docs - self.k + 1)) + (1 - self.eps_greedy))).to(torch.float)
                self.doc_per_rank_prob_det[qids_mapped[i], truncated_rankings] = (self.alpha_tensor_exp[self.k-1] * (self.eps_greedy *(1/(num_docs - self.k + 1)))).to(torch.float)
                rel_qid_k = torch.gather(rel_labels_qid, 1, kth_sample_index.unsqueeze(0))
                click_prob = self.alpha_tensor_exp.to(self.device)[self.k-1] * rel_qid_k  + self.noise
                clicks = torch.bernoulli(click_prob).squeeze(0).float().to('cpu')
                self.qid_clicks[:, :, self.k-1].index_put_((ix, kth_sample), clicks, accumulate=True)
                #self.qid_clicks[qids_mapped[i], kth_sample, self.k-1] += clicks.to('cpu')
        return None
        


        
        

