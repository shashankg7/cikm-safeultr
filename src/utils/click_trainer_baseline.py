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


def ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, clip=0.0001, mode='train'):
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    clip = torch.tensor(clip).to(device)
    doc_prob_rank = doc_prob_rank.to(device)
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
    #weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    weights_per_rank = (1/(torch.arange(label_size[2])+1))
    weights_per_rank = weights_per_rank.to(device)
    batch_size = sampled_rankings.shape[0]
    
    weights_per_rank[k:] = 0.
    grad_weight = (rel_labels * weights_per_rank).sum(-1)
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    # weigh by the log-scores - REINFORCE trick
    if mode == 'train':
        logging_policy = torch.ones(batch_size, num_samples).to(device)
        logging_policy.requires_grad = False
        ix = torch.repeat_interleave(torch.arange(batch_size), num_samples)
        for i in range(k):
            iy = sampled_rankings[:, :, i].reshape(batch_size * num_samples)
            rank_prob = torch.gather(input=doc_prob_rank[...,i].reshape(batch_size * label_size[2]), dim=0, index=iy).reshape(batch_size, num_samples)
            logging_policy *= rank_prob
        logging_policy = torch.maximum(logging_policy, clip)
        grad_weight = grad_weight/logging_policy
        #grad_weight = (ips_weight * weights_per_rank)
        #out, _ = torch.max(grad_weight, dim=-1)
        #grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
        grad_weight.nan_to_num_()
        obj = grad_weight
        cv = grad_weight.mean(-1).reshape(-1, 1)     
        obj = log_scores * (obj/dcg_norm)
        # aggregate scores
        obj = -torch.mean(torch.sum(obj, dim=-1)) #+ 1e-2 * entropy_reg
    else:
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(device)
        weights_per_rank[k:] = 0.
        grad_weight = (rel_labels * weights_per_rank).sum(-1)
        obj = grad_weight
        obj = torch.mean(obj/dcg_norm, dim=-1)
    return obj



def ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, eta, clip=0.0001, mode='train'):
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    clip = torch.tensor(clip).to(device)
    doc_prob_rank = doc_prob_rank.to(device)
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
    #weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    weights_per_rank = (1/(torch.arange(label_size[2])+1))**eta
    weights_per_rank = weights_per_rank.to(device)
    batch_size = sampled_rankings.shape[0]
    
    weights_per_rank[k:] = 0.
    grad_weight = (rel_labels * weights_per_rank).sum(-1)
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    # weigh by the log-scores - REINFORCE trick
    if mode == 'train':
        logging_policy = torch.ones(batch_size, num_samples).to(device)
        logging_policy.requires_grad = False
        ix = torch.repeat_interleave(torch.arange(batch_size), num_samples)
        for i in range(k):
            iy = sampled_rankings[:, :, i].reshape(batch_size * num_samples)
            rank_prob = torch.gather(input=doc_prob_rank[...,i].reshape(batch_size * label_size[2]), dim=0, index=iy).reshape(batch_size, num_samples)
            logging_policy *= rank_prob
        logging_policy = torch.maximum(logging_policy, clip)
        grad_weight = grad_weight/logging_policy
        #grad_weight = (ips_weight * weights_per_rank)
        #out, _ = torch.max(grad_weight, dim=-1)
        #grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
        grad_weight.nan_to_num_()
        obj = grad_weight
        cv = grad_weight.mean(-1).reshape(-1, 1)     
        obj = log_scores * (obj/dcg_norm-cv)
        # aggregate scores
        obj = -torch.mean(torch.sum(obj, dim=-1)) #+ 1e-2 * entropy_reg
    else:
        weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
        weights_per_rank = weights_per_rank.to(device)
        weights_per_rank[k:] = 0.
        grad_weight = (rel_labels * weights_per_rank).sum(-1)
        obj = grad_weight
        obj = torch.mean(obj/dcg_norm, dim=-1)
    return obj


def risk_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, mode, clip=0.0001):
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    doc_prob_rank = doc_prob_rank.to(device)
    clip = torch.tensor(clip).to(device)
    clip_policy = torch.tensor(1e-6).to(device)
    batch_size = doc_prob_rank.shape[0]
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
    policy_prob = torch.exp(log_scores.detach().clone())
    #weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    weights_per_rank = (1/(torch.arange(label_size[2])+1))
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank.requires_grad = False
    logging_policy = torch.ones(batch_size, num_samples).to(device)
    logging_policy.requires_grad = False
    ix = torch.repeat_interleave(torch.arange(batch_size), num_samples)
    for i in range(k):
        iy = sampled_rankings[:, :, i].reshape(batch_size * num_samples)
        rank_prob = torch.gather(input=doc_prob_rank[...,i].reshape(batch_size * label_size[2]), dim=0, index=iy).reshape(batch_size, num_samples)
        logging_policy *= rank_prob
    
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    logging_policy = torch.maximum(logging_policy, clip)
    #policy_prob = torch.maximum(policy_prob, clip_policy)
    grad_weight = (policy_prob /logging_policy)
    #grad_weight = (ips_weight * weights_per_rank)
    #out, _ = torch.max(grad_weight, dim=-1)
    #grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
    if mode == 'train':
        cv = grad_weight.mean(-1).reshape(-1, 1)     
        obj = log_scores * ( grad_weight - cv) 
        obj = obj.mean(-1).mean(-1)
        reg = obj
    else:
        reg = grad_weight.mean(-1).mean(-1)
    return reg


def risk_obj1(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, mode, clip=0.0001):
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    doc_prob_rank = doc_prob_rank.to(device)
    clip = torch.tensor(clip).to(device)
    clip_policy = torch.tensor(1e-6).to(device)
    batch_size = doc_prob_rank.shape[0]
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
    policy_prob = torch.exp(log_scores.detach().clone())
    #weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    weights_per_rank = (1/(torch.arange(label_size[2])+1))
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank.requires_grad = False
    logging_policy = torch.ones(batch_size, num_samples).to(device)
    logging_policy.requires_grad = False
    ix = torch.repeat_interleave(torch.arange(batch_size), num_samples)
    for i in range(k):
        iy = sampled_rankings[:, :, i].reshape(batch_size * num_samples)
        rank_prob = torch.gather(input=doc_prob_rank[...,i].reshape(batch_size * label_size[2]), dim=0, index=iy).reshape(batch_size, num_samples)
        logging_policy *= rank_prob
    
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    logging_policy = torch.maximum(logging_policy, clip)
    #policy_prob = torch.maximum(policy_prob, clip_policy)
    grad_weight = (policy_prob /logging_policy)
    reg_denom = torch.sqrt(torch.mean((policy_prob.detach()/logging_policy.detach()).mean(-1)))
    grad_weight /= reg_denom
    #grad_weight = (ips_weight * weights_per_rank)
    #out, _ = torch.max(grad_weight, dim=-1)
    #grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
    if mode == 'train':
        cv = grad_weight.mean(-1).reshape(-1, 1)     
        obj = log_scores * ( grad_weight - cv) 
        obj = obj.mean(-1).mean(-1)
        reg = obj
    else:
        reg = grad_weight.mean(-1).mean(-1)
    return reg


def trainer(num_queries, num_samples, k, lr, optimizer, alpha, beta,\
             meta_dir, device, train_dataloader, val_dataloader, test_dataloader, wandb,\
                  ips_file, eta, epochs=10, get_exposure=False,\
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
                doc_prob_rank = batch['alpha']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank,eta, clip=1e-5, mode='train')
                wandb.log({
                        "utility loss": obj.item()})
                optimizer.zero_grad()
                obj.backward(retain_graph=False)
                optimizer.step()
            doc_scorer.eval()
            # check for early stopping
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                doc_prob_rank = batch['alpha']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, eta, clip=1e-7, mode='test')
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
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, eta, clip=0.0001, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        print(np.mean(np.hstack(test_dcg)))
        ips_file.write(str(np.mean(np.hstack(test_dcg))))



def trainer_risk(num_queries, num_samples, k, lr, optimizer, alpha, beta,\
             meta_dir, device, train_dataloader, val_dataloader, train_dataloader_risk, test_dataloader, wandb,\
                 risk_file, epochs=10, get_exposure=False,\
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
        optimizer1 = optim.Adam(doc_scorer.parameters(), lr=lr, weight_decay=0)
        #optimizer = optim.SGD(doc_scorer.parameters(), lr=0.01, weight_decay=1e-5)
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
                doc_prob_rank = batch['alpha']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, clip=0.0001, mode='train')
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                wandb.log({
                        "utility loss": obj.item()})
                optimizer.zero_grad()
                obj.backward(retain_graph=False)
                optimizer.step()
                #for j in range(10):
                if (x + 1)%alternate_freq == 0:
                    for j, batch in enumerate(train_dataloader_risk):
                        #torch.autograd.set_detect_anomaly(True)
                        rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                        doc_prob_rank = batch['alpha']
                        #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                        #obj =  (10/np.log2(num_queries))* risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                        obj =  np.power(0.9, np.sqrt(num_queries))* risk_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, clip=0.0001)
                        #obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                        #reg = (1/num_queries) * obj 
                        #reg = (10/np.sqrt(num_queries)) *obj 
                        reg = obj 
                        wandb.log({
                            "risk loss": reg.item()})
                        #print(obj)
                        #print(log_scores.sum())
                        #print(obj)
                        optimizer1.zero_grad()
                        obj.backward()
                        optimizer1.step()
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
                doc_prob_rank = batch['alpha']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, clip=0.0001, mode='test')
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
            #doc_prob_rank = batch['alpha']
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, clip=0.0001, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write( str(np.mean(np.hstack(test_dcg))))




def trainer_risk_linear(num_queries, num_samples, k, lr, optimizer, alpha, beta,\
             meta_dir, device, train_dataloader, val_dataloader, test_dataloader, wandb,\
                 risk_file, delta, eta, epochs=10, get_exposure=False,\
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
        optimizer1 = optim.Adam(doc_scorer.parameters(), lr=lr, weight_decay=0)
        #optimizer = optim.SGD(doc_scorer.parameters(), lr=0.01, weight_decay=1e-5)
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
                doc_prob_rank = batch['alpha']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, eta, clip=0.0001, mode='train')
                wandb.log({
                        "ips loss": obj.item()})
                #doc_prob_rank = batch['alpha']
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                #obj =  (10/np.log2(num_queries))* risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                reg =  delta * np.sqrt(1/num_queries) * risk_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, mode='train', clip=0.0001)
                #obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                #reg = (1/num_queries) * obj 
                #reg = (10/np.sqrt(num_queries)) *obj 
                wandb.log({
                    "risk loss": reg.item()})
                obj += reg
                optimizer.zero_grad()
                obj.backward(retain_graph=False)
                optimizer.step()
                #for j in range(10):
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                doc_prob_rank = batch['alpha']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, eta, clip=0.0001, mode='test')
                #rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                #alpha = batch['alpha']
                reg = -delta * np.sqrt(1/num_queries) * risk_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, mode='test', clip=0.0001)
                obj += reg
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
            #doc_prob_rank = batch['alpha']
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, eta, clip=0.0001, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write( str(np.mean(np.hstack(test_dcg))))


def trainer_risk_linear_sqrt(num_queries, num_samples, k, lr, optimizer, alpha, beta,\
             meta_dir, device, train_dataloader, val_dataloader, test_dataloader, wandb,\
                 risk_file, delta, eta, epochs=10, get_exposure=False,\
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
        optimizer1 = optim.Adam(doc_scorer.parameters(), lr=lr, weight_decay=0)
        #optimizer = optim.SGD(doc_scorer.parameters(), lr=0.01, weight_decay=1e-5)
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
                doc_prob_rank = batch['alpha']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank,eta, clip=0.0001, mode='train')
                wandb.log({
                        "ips loss": obj.item()})
                doc_prob_rank = batch['alpha']
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                #obj =  (10/np.log2(num_queries))* risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                reg =  delta * np.sqrt(1/num_queries) * risk_obj1(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, mode='train',clip=0.0001)
                #obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                #reg = (1/num_queries) * obj 
                #reg = (10/np.sqrt(num_queries)) *obj 
                wandb.log({
                    "risk loss": reg.item()})
                obj += reg
                optimizer.zero_grad()
                obj.backward(retain_graph=False)
                optimizer.step()
                #for j in range(10):
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                doc_prob_rank = batch['alpha']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, eta, clip=0.0001, mode='test')
                #rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                #alpha = batch['alpha']
                reg = -delta * np.sqrt(1/num_queries) * risk_obj1(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, mode='test',clip=0.0001)
                obj += reg
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
            #doc_prob_rank = batch['alpha']
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, doc_prob_rank, eta, clip=0.0001, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write( str(np.mean(np.hstack(test_dcg))))






