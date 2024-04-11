'''
Trains click model in plain pytorch
'''

from cmath import exp
import os, json, pdb
import pandas as pd
import numpy as np


import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
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


def ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train'):
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    doc_feats = doc_feats.squeeze(1).float()
    doc_scores = doc_scorer(doc_feats).squeeze(-1)
    # generate samples from PL-model
    ranking_scores, sampled_rankings = pl_sampler.sample(doc_scores, mask)
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
    weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    weights_per_rank = weights_per_rank.to(device)
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    obj = rel_labels * weights_per_rank
    #target = torch.ones_like(doc_scores)
    # take top-k elements
    #pdb.set_trace()
    obj = obj[:, :, :k].sum(-1)
    cv = obj.mean(-1).reshape(-1, 1)
    # weigh by the log-scores - REINFORCE trick
    if mode == 'train':
        obj = log_scores * (obj/dcg_norm-cv)
        # aggregate scores
        obj = -torch.mean(torch.sum(obj, dim=-1)) #+ 1e-2 * entropy_reg
    else:
        obj = torch.mean(obj/dcg_norm, dim=-1)
    return obj


def risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k):
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    alpha = alpha.to(device)
    batch_size = alpha.shape[0]
    doc_feats = doc_feats.squeeze(1).float()
    doc_scores = doc_scorer(doc_feats).squeeze(-1)
    # generate samples from PL-model
    ranking_scores, sampled_rankings = pl_sampler.sample(doc_scores, mask)
    size = doc_scores.shape
    num_rankings = sampled_rankings.shape[1]
    max_cand_size = sampled_rankings.shape[2]
    # expand rel_labels to match sampled ranking scores tensor size
    label_size = rel_labels.shape
    rel_labels = rel_labels.expand(label_size[0], num_samples, label_size[2])
    rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
    doc_scores = doc_scores.unsqueeze(1).expand(size[0], num_samples, size[1])
    doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
    log_scores, _   = pl_sampler.log_scores(doc_scores, mask, k=k)
    weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank.requires_grad = False
    prop_counter = torch.zeros(batch_size, max_cand_size).to(device)
    prop_counter.requires_grad = False
    ix = torch.repeat_interleave(torch.arange(batch_size), num_rankings)
    for i in range(k):
        iy = sampled_rankings[:, :, i].reshape(batch_size * num_rankings)
        prop_counter.index_put_((ix, iy), torch.tensor(1/torch.log2(torch.tensor(i+2))), accumulate=True)
        #prop_counter.index_put_((ix, iy), torch.tensor(1/torch.tensor(i+1)), accumulate=True)
    prop_counter /= num_rankings
    #reg_denom = torch.sqrt(torch.mean((torch.square(prop_counter/alpha) * alpha).sum(-1)))
    alpha = alpha.unsqueeze(1).expand(batch_size, num_samples, -1)
    alpha = torch.gather(alpha, 2, sampled_rankings)
    prop_counter = prop_counter.unsqueeze(1).expand(label_size[0], num_samples, label_size[2])
    prop_counter = torch.gather(prop_counter, 2, sampled_rankings)
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    ips_weight = prop_counter/alpha
    #ips_weight/ips_weight.sum(-1).reshape(ips_weight.shape[0], ips_weight.shape[1], 1)
    ips_weight.nan_to_num_()
    #torch.clamp_(ips_weight, max=1)
    weights_per_rank[k:] = 0.
    #ips_weight = ips_weight[:, : , :k]
    grad_weight = (ips_weight * weights_per_rank).sum(-1)
    #out, _ = torch.max(grad_weight, dim=-1)
    grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
    #grad_weight = grad_weight * log_scores
    #reg = torch.mean(((ips_weight * weights_per_rank).sum(-1) * log_scores).sum(-1))
    cv = grad_weight.mean(-1).reshape(-1, 1)     
    obj = log_scores * (grad_weight - cv)   
    #reg = -torch.mean((grad_weight * log_scores).sum(-1)) #- 0.01 * entropy_loss(doc_scores, target)
    #reg = torch.mean(grad_weight.sum(-1)) #-  entropy_loss(doc_scores, target)
    reg =  torch.mean(obj.sum(-1)) #-  entropy_loss(doc_scores, target)
    return reg


def trainer(num_queries, num_samples, k, lr, optimizer, alpha, beta,\
             meta_dir, device, train_dataloader, val_dataloader, test_dataloader, wandb, hp_file_ips,\
                  epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        doc_scorer = DocScorer(**MLP_args)
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
        #optimizer = optim.SGD(doc_scorer.parameters(), lr=lr, weight_decay=1e-5)
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
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                optimizer.zero_grad()
                obj.backward(retain_graph=False)
                optimizer.step()
            doc_scorer.eval()
            # check for early stopping
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='test')
                val_dcg.append(obj.to('cpu').numpy())
            final_val_dcg = np.mean(np.hstack(val_dcg))
            if best_score is None:
                best_score = final_val_dcg
            else:
                if final_val_dcg > best_score:
                    best_score = final_val_dcg
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break
        print(best_score)
        print(hp_file_ips)
        hp_file_ips.write(str(best_score) + '\n')



def trainer_risk(num_queries, num_samples, k, lr, optimizer, alpha, beta,\
             meta_dir, device, train_dataloader, val_dataloader, train_dataloader_risk, test_dataloader, wandb, hp_file_risk,\
                  epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        doc_scorer = DocScorer(**MLP_args)
        doc_scorer = doc_scorer.to(device)
        k = k
        lr = lr
        reg_weight = 0.
        util_weight = 1.
        optimizer = optimizer
        #torch.autograd.set_detect_anomaly(True)
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
        #optimizer = optim.SGD(doc_scorer.parameters(), lr=lr, weight_decay=1e-5)
        #scheduler = ExponentialLR(optimizer, gamma=0.7)
        #optimizer = optim.SGD(doc_scorer.parameters(), lr=1e-2, weight_decay=1e-4)
        doc_scorer = doc_scorer.to(device)
        #optimizer1 = optim.Adam(doc_scorer.parameters(), lr=0.00001, weight_decay=1e-3)
        best_score = None
        counter = 0
        patience = 3
        alternate_freq = 1
        risk_steps = 1
        for epoch in range(epochs):
            val_dcg = []
            # for g in optimizer.param_groups:
            #     g['lr'] = g['lr'] * (10/np.sqrt(num_queries))
            doc_scorer.train()
            # for g in optimizer.param_groups:
            #     g['lr'] = 1e-3
            for x, batch in enumerate(train_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                optimizer.zero_grad()
                obj.backward(retain_graph=False)
                optimizer.step()
                
                #for j in range(10):
                if (x + 1)%alternate_freq == 0:
                    for j, batch in enumerate(train_dataloader_risk):
                        #torch.autograd.set_detect_anomaly(True)
                        rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                        alpha = batch['alpha']
                        #obj =  (10/np.sqrt(num_queries)) * risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                        obj =  (10/np.sqrt(num_queries))* risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                        #obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                        #reg = (1/num_queries) * obj 
                        #reg = (10/np.sqrt(num_queries)) *obj 
                        reg = obj 
                        #print(obj)
                        #print(log_scores.sum())
                        #print(obj)
                        optimizer.zero_grad()
                        obj.backward()
                        optimizer.step()
                        # aggregate scores
                        # obj = -util_weight * torch.mean(torch.sum(obj, dim=-1)) + reg_weight * reg
                        # optimizer.zero_grad()
                        # obj.backward()
                        # optimizer.step()
                        if j > risk_steps:
                            break 
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='test')
                val_dcg.append(obj.to('cpu').numpy())
            final_val_dcg = np.mean(np.hstack(val_dcg))
            if best_score is None:
                best_score = final_val_dcg
            else:
                if final_val_dcg > best_score:
                    best_score = final_val_dcg
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break
        print(best_score)
        hp_file_risk.write(str(best_score) + '\n')


def trainer_risk_alternate(num_queries, num_samples, k, lr, optimizer, alpha, beta,\
             meta_dir, device, train_dataloader, val_dataloader, train_dataloader_risk, test_dataloader, wandb, hp_file_risk,\
                  epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        doc_scorer = DocScorer(**MLP_args)
        doc_scorer = doc_scorer.to(device)
        k = k
        lr = lr
        reg_weight = 0.
        util_weight = 1.
        optimizer = optimizer
        #torch.autograd.set_detect_anomaly(True)
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
        #optimizer = optim.SGD(doc_scorer.parameters(), lr=lr, weight_decay=1e-5)
        #scheduler = ExponentialLR(optimizer, gamma=0.7)
        #optimizer = optim.SGD(doc_scorer.parameters(), lr=1e-2, weight_decay=1e-4)
        doc_scorer = doc_scorer.to(device)
        #optimizer1 = optim.Adam(doc_scorer.parameters(), lr=0.00001, weight_decay=1e-3)
        best_score = None
        counter = 0
        patience = 3
        alternate_freq = 1
        risk_steps = 1
        for epoch in range(epochs):
            val_dcg = []
            # for g in optimizer.param_groups:
            #     g['lr'] = g['lr'] * (10/np.sqrt(num_queries))
            doc_scorer.train()
            # for g in optimizer.param_groups:
            #     g['lr'] = 1e-3
            for x, batch in enumerate(train_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                optimizer.zero_grad()
                obj.backward(retain_graph=False)
                optimizer.step()
                
            for j, batch in enumerate(train_dataloader_risk):
                #torch.autograd.set_detect_anomaly(True)
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                #obj =  (10/np.sqrt(num_queries)) * risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                obj =  (10/np.sqrt(num_queries))* risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                #obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                #reg = (1/num_queries) * obj 
                #reg = (10/np.sqrt(num_queries)) *obj 
                reg = obj 
                #print(obj)
                #print(log_scores.sum())
                #print(obj)
                optimizer.zero_grad()
                obj.backward()
                optimizer.step()
                # aggregate scores
                # obj = -util_weight * torch.mean(torch.sum(obj, dim=-1)) + reg_weight * reg
                # optimizer.zero_grad()
                # obj.backward()
                # optimizer.step()
                #if j > risk_steps:
                #    break 
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='test')
                val_dcg.append(obj.to('cpu').numpy())
            final_val_dcg = np.mean(np.hstack(val_dcg))
            if best_score is None:
                best_score = final_val_dcg
            else:
                if final_val_dcg > best_score:
                    best_score = final_val_dcg
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break
        print(best_score)
        hp_file_risk.write(str(best_score) + '\n')







