'''
Trains click model in plain pytorch
'''

from cmath import exp
import os, json, pdb, copy
import pandas as pd
import numpy as np


import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch
import torch.nn as nn
from src.models.nnmodel import DocScorer
from src.utils.PlackettLuce import PlackettLuceModel
import wandb



def ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train'):
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    doc_feats = doc_feats.squeeze(1).float()
    doc_scores = doc_scorer(doc_feats).squeeze(-1)
    # generate samples from PL-model
    ranking_scores, sampled_rankings = pl_sampler.sample(doc_scores, mask)
    size = doc_scores.shape
    # expand rel_labels to match sampled ranking scores tensor size
    label_size = rel_labels.shape
    rel_labels = rel_labels.expand(label_size[0], num_samples, label_size[2])
    rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
    doc_scores = doc_scores.unsqueeze(1).expand(size[0], num_samples, size[1])
    doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
    log_scores, _   = pl_sampler.log_scores(doc_scores, mask, k=k)
    weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    #weights_per_rank = (1/(torch.log2arange(label_size[2])+2))
    weights_per_rank = weights_per_rank.to(device)
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    obj = rel_labels * weights_per_rank
    # take top-k elements
    #pdb.set_trace()
    obj = obj[:, :, :k].sum(-1)
    # weigh by the log-scores - REINFORCE trick
    if mode == 'train':
        obj = log_scores * (obj/dcg_norm)
        # aggregate scores
        obj = -torch.mean(torch.sum(obj, dim=-1))
    else:
        obj = torch.mean(obj/dcg_norm, dim=-1)
    return obj


def ips_obj_cv(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='train'):
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    alpha = alpha.to(device)
    doc_feats = doc_feats.squeeze(1).float()
    doc_scores = doc_scorer(doc_feats).squeeze(-1)
    # generate samples from PL-model
    ranking_scores, sampled_rankings = pl_sampler.sample(doc_scores, mask)
    num_rankings = sampled_rankings.shape[1]
    max_cand_size = sampled_rankings.shape[2]
    size = doc_scores.shape
    prob = nn.functional.softmax(doc_scores, dim=-1)
    log_scores = nn.functional.log_softmax(doc_scores, dim=-1)
    entropy_reg = -(prob * log_scores).sum(-1).mean()
    # expand rel_labels to match sampled ranking scores tensor size
    label_size = rel_labels.shape
    rel_labels = rel_labels.expand(label_size[0], num_samples, label_size[2])
    rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
    doc_scores = doc_scores.unsqueeze(1).expand(size[0], num_samples, size[1])
    doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
    log_scores, _   = pl_sampler.log_scores(doc_scores, mask, k=k)
    #weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    batch_size = label_size[0]
    weights_per_rank = (1/(torch.arange(label_size[2])+1))**eta
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank[k:] = 0.
    weights_per_rank.requires_grad = False
    # propensity ratio compute
    prop_counter = torch.zeros(batch_size, max_cand_size)
    prop_counter.requires_grad = False
    ix = torch.repeat_interleave(torch.arange(batch_size), num_rankings)
    for i in range(k):
        iy = sampled_rankings[:, :, i].reshape(batch_size * num_rankings)
        #w = torch.zeros_like(iy)
        #w[:] = torch.tensor(1/torch.log2(torch.tensor(i+2)))
        prop_counter.index_put_((ix, iy), (torch.tensor(1/(i+1))**eta).float(), accumulate=True)
        #prop_counter.index_put_((ix, iy), torch.tensor(1/np.log2(i+2)).float(), accumulate=True)
    prop_counter /= num_rankings
    prop_counter = prop_counter.to(device)
    ips_weight = prop_counter/alpha
    ips_weight = ips_weight.unsqueeze(1).expand(batch_size, num_samples, -1)
    ips_weight = torch.gather(ips_weight, 2, sampled_rankings)
    grad_clip = (ips_weight < 1.2).int()
    grad_clip[grad_clip==0] = -0.1
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    obj = rel_labels * weights_per_rank * grad_clip
    #target = torch.ones_like(doc_scores)
    # take top-k elements
    #pdb.set_trace()
    obj = obj.sum(-1)
    cv = obj.mean(-1).reshape(-1, 1)
    # weigh by the log-scores - REINFORCE trick
    if mode == 'train':
        obj = log_scores * (obj/dcg_norm-cv)
        #obj = log_scores * (obj/dcg_norm)
        # aggregate scores
        obj = -torch.mean(torch.mean(obj, dim=-1)) #+ 1e-2 * entropy_reg
    else:
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        #weights_per_rank = (1/(torch.arange(label_size[2])+1))
        weights_per_rank = weights_per_rank.to(device)
        weights_per_rank[k:] = 0.
        # LTR metric/objective func - DCG with current weights_per_rank
        # TO-DO: Make it configurable, user should be able to specify the metric
        obj = rel_labels * weights_per_rank
        #target = torch.ones_like(doc_scores)
        # take top-k elements
        #pdb.set_trace()
        obj = obj.sum(-1)
        obj = torch.mean(obj/dcg_norm, dim=-1)
    return obj


def trainer(num_queries, num_samples, k, lr, optimizer, alpha, beta,\
             meta_dir, device, train_dataloader, val_dataloader, test_dataloader, wandb,\
                  risk_file, eta, logging_policy, epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        # initialize the model randomly
        doc_scorer = DocScorer(**MLP_args)
        # initialize with the logging policy model
        #doc_scorer = logging_policy.doc_scorer
        doc_scorer = doc_scorer.to(device)
        k = k
        lr = lr
        reg_weight = 0.
        util_weight = 1.
        optimizer = optimizer
        torch.autograd.set_detect_anomaly(True)
        beta_tensor = torch.tensor(beta).to(device)
        alpha_tensor = torch.tensor(alpha).to(device)
        #output_dir = out_dir
        meta_dir = meta_dir
        pl_sampler = PlackettLuceModel(num_samples)
        # weights per rank, for computing the LTR metric
        weights_per_rank = 1/np.log2(np.arange(k)+2)
        # get_exposure flag tells whether we are in click simulation mode or in expected exposure computation mode
        get_exposure = get_exposure
        # running average estimate of the regularization term
        running_estimate = torch.tensor(0.)
        running_estimate = running_estimate.to(device)
        running_estimate_len = torch.tensor(0.).to(device)
        optimizer = optim.Adam(doc_scorer.parameters(), lr=lr, weight_decay=1e-5)
        #optimizer = optim.SGD(doc_scorer.parameters(), lr=0.01, weight_decay=1e-5)
        doc_scorer = doc_scorer.to(device)
        #optimizer1 = optim.Adam(doc_scorer.parameters(), lr=0.00001, weight_decay=1e-3)
        best_score = None
        counter = 0
        patience = 3

        for epoch in range(epochs):
            val_dcg = []
            doc_scorer.train()
            # for g in optimizer.param_groups:
            #     g['lr'] = g['lr'] * (10/np.sqrt(num_queries))
            for i, batch in enumerate(train_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                obj = ips_obj_cv(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='train')
                wandb.log({
                        "utility loss": obj.item()})
                optimizer.zero_grad()
                obj.backward(retain_graph=False)
                optimizer.step()
            doc_scorer.eval()
            # check for early stopping
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                obj = ips_obj_cv(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='test')
                val_dcg.append(obj.to('cpu').numpy())
            final_val_dcg = np.mean(np.hstack(val_dcg))
            wandb.log({
                        "val-dcg@5": final_val_dcg})
            if best_score is None:
                best_score = final_val_dcg
            else:
                if final_val_dcg > best_score:
                    best_score = final_val_dcg
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        wandb.log({
                        "epoch": epoch})
                        break
        doc_scorer.eval()            
        for i, batch in enumerate(test_dataloader):
            rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
            #alpha = batch['alpha']
            obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write(str(np.mean(np.hstack(test_dcg))))













