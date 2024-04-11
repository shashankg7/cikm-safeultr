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


divergence_weight = 0.01

def ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, display_mask=None, mode='train'):
    # if display_mask is not None:
    #     rel_labels *= display_mask.unsqueeze(1)
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
    #weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    weights_per_rank = torch.tensor(alpha_w + beta_w)
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank[k:] = 0.
    if weights_per_rank.shape[0] > label_size[2]:
        weights_per_rank = weights_per_rank[:label_size[2]]
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
        obj = -torch.mean(torch.mean(obj, dim=-1))
    else:
        obj = torch.mean(obj, dim=-1)
    return obj


def ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, display_mask=None, mode='train'):
    entropy_loss = nn.CrossEntropyLoss()
    # if display_mask is not None:
    #     rel_labels *= display_mask.unsqueeze(1)
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
    #weights_per_rank = (1/(torch.arange(label_size[2])+1))**eta
    weights_per_rank = torch.tensor(alpha_w + beta_w)
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank[k:] = 0.
    if weights_per_rank.shape[0] > label_size[2]:
        weights_per_rank = weights_per_rank[:label_size[2]]
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    obj = rel_labels * weights_per_rank
    #target = torch.ones_like(doc_scores)
    # take top-k elements
    #pdb.set_trace()
    obj = obj.sum(-1)
    cv = obj.mean(-1).reshape(-1, 1)
    # weigh by the log-scores - REINFORCE trick
    if mode == 'train':
        obj = log_scores * (obj/dcg_norm-cv)
        # obj = log_scores * (obj/dcg_norm)
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


def ips_risk_obj_cv(N, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_w, beta_w, mode='train', risk=False):
    weights_per_rank = torch.tensor(alpha_w + beta_w)
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank[k:] = 0.
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    doc_feats = doc_feats.squeeze(1).float()
    doc_scores = doc_scorer(doc_feats).squeeze(-1)
    # generate samples from PL-model
    ranking_scores, sampled_rankings = pl_sampler.sample(doc_scores, mask)
    ips, reg_denom = get_propensity(device, doc_scores, alpha, mask, pl_sampler, k, weights_per_rank, sampled_rankings, reuse_sampled_rankings=False)
    ips /= reg_denom
    div_weight = np.sqrt((1-divergence_weight)/(divergence_weight*N))
    ips *= div_weight
    size = doc_scores.shape
    prob = nn.functional.softmax(doc_scores, dim=-1)
    log_scores = nn.functional.log_softmax(doc_scores, dim=-1)
    entropy_reg = -(prob * log_scores).sum(-1).mean()
    # expand rel_labels to match sampled ranking scores tensor size
    label_size = rel_labels.shape
    # modify reward to adjust for divergence
    #rel_labels -= ips.unsqueeze(1)
    rel_labels = rel_labels.expand(label_size[0], num_samples, label_size[2])
    rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
    doc_scores = doc_scores.unsqueeze(1).expand(size[0], num_samples, size[1])
    doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
    log_scores, _   = pl_sampler.log_scores(doc_scores, mask, k=k)
    #weights_per_rank = (1/(torch.arange(label_size[2])+1))**eta
    if weights_per_rank.shape[0] > label_size[2]:
        weights_per_rank = weights_per_rank[:label_size[2]]
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    obj = rel_labels * weights_per_rank
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


def dr_obj_cv(N, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w, reg_model, mode='train', risk=False):
    weights_per_rank = torch.tensor(alpha_w + beta_w)
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank[k:] = 0.
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    rho = rho.to(device)
    alpha_freq = alpha_freq.to(device)
    display_mask = display_mask.to(device)
    doc_feats = doc_feats.squeeze(1).float()
    doc_scores = doc_scorer(doc_feats).squeeze(-1)
    # get regression scores from the trained regression model
    reg_scores = torch.sigmoid(reg_model(doc_feats).squeeze(-1))
    # generate samples from PL-model
    ranking_scores, sampled_rankings = pl_sampler.sample(doc_scores, mask)
    size = doc_scores.shape
    prob = nn.functional.softmax(doc_scores, dim=-1)
    log_scores = nn.functional.log_softmax(doc_scores, dim=-1)
    entropy_reg = -(prob * log_scores).sum(-1).mean()
    # expand rel_labels to match sampled ranking scores tensor size
    label_size = rel_labels.shape
    # PPO gradient mask. mask for positive reward
    # modify reward to adjust for divergence
    #rel_labels -= ips.unsqueeze(1)
    # modify reward for DR estimator
    dr_reward = (reg_scores * (1-alpha_freq/rho))#*display_mask
    #dr_reward = reg_scores - (reg_scores * (alpha_freq/rho))*display_mask
    rel_labels += dr_reward.unsqueeze(1)
    rel_labels = rel_labels.expand(label_size[0], num_samples, label_size[2])
    rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
    doc_scores = doc_scores.unsqueeze(1).expand(size[0], num_samples, size[1])
    doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
    log_scores, _   = pl_sampler.log_scores(doc_scores, mask, k=k)
    #weights_per_rank = (1/(torch.arange(label_size[2])+1))**eta
    if weights_per_rank.shape[0] > label_size[2]:
        weights_per_rank = weights_per_rank[:label_size[2]]
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    obj = rel_labels * weights_per_rank
    #target = torch.ones_like(doc_scores)
    # take top-k elements
    #pdb.set_trace()
    obj = obj.sum(-1)
    cv = obj.mean(-1).reshape(-1, 1)
    # weigh by the log-scores - REINFORCE trick
    if mode == 'train':
        obj = log_scores * (obj/dcg_norm-cv)
        # obj = log_scores * (obj/dcg_norm)
        # aggregate scores
        obj = -torch.mean(torch.mean(obj, dim=-1)) #+ 1e-2 * entropy_reg
    else:
        obj = obj.mean(-1)
    return obj


def dr_risk_obj_cv(N, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w, reg_model, mode='train', risk=False):
    weights_per_rank = torch.tensor(alpha_w + beta_w)
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank[k:] = 0.
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    rho = rho.to(device)
    alpha_freq = alpha_freq.to(device)
    display_mask = display_mask.to(device)
    doc_feats = doc_feats.squeeze(1).float()
    doc_scores = doc_scorer(doc_feats).squeeze(-1)
    # get regression scores from the trained regression model
    reg_scores = torch.sigmoid(reg_model(doc_feats).squeeze(-1))
    # generate samples from PL-model
    ranking_scores, sampled_rankings = pl_sampler.sample(doc_scores, mask)
    ips, reg_denom = get_propensity(device, doc_scores, alpha, mask, pl_sampler, k, weights_per_rank, sampled_rankings, reuse_sampled_rankings=False)
    ips /= reg_denom
    div_weight = np.sqrt((1-divergence_weight)/(divergence_weight*N))
    ips *= div_weight
    size = doc_scores.shape
    prob = nn.functional.softmax(doc_scores, dim=-1)
    log_scores = nn.functional.log_softmax(doc_scores, dim=-1)
    entropy_reg = -(prob * log_scores).sum(-1).mean()
    # expand rel_labels to match sampled ranking scores tensor size
    label_size = rel_labels.shape
    # PPO gradient mask. mask for positive reward
    # modify reward for DR estimator
    dr_reward = (reg_scores * (1-alpha_freq/rho))#*display_mask
    #dr_reward = reg_scores - (reg_scores * (alpha_freq/rho))*display_mask
    rel_labels += dr_reward.unsqueeze(1)
    # modify reward to adjust for divergence
    rel_labels -= ips.unsqueeze(1)
    rel_labels = rel_labels.expand(label_size[0], num_samples, label_size[2])
    rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
    doc_scores = doc_scores.unsqueeze(1).expand(size[0], num_samples, size[1])
    doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
    log_scores, _   = pl_sampler.log_scores(doc_scores, mask, k=k)
    #weights_per_rank = (1/(torch.arange(label_size[2])+1))**eta
    if weights_per_rank.shape[0] > label_size[2]:
        weights_per_rank = weights_per_rank[:label_size[2]]
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    obj = rel_labels * weights_per_rank
    #target = torch.ones_like(doc_scores)
    # take top-k elements
    #pdb.set_trace()
    obj = obj.sum(-1)
    cv = obj.mean(-1).reshape(-1, 1)
    # weigh by the log-scores - REINFORCE trick
    if mode == 'train':
        obj = log_scores * (obj/dcg_norm-cv)
        # obj = log_scores * (obj/dcg_norm)
        # aggregate scores
        obj = -torch.mean(torch.mean(obj, dim=-1)) #+ 1e-2 * entropy_reg
    else:
        obj = obj.mean(-1)
    return obj


def dr_ppo_obj_cv(N, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w, reg_model, mode='train', risk=False):
    weights_per_rank = torch.tensor(alpha_w + beta_w)
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank[k:] = 0.
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    rho = rho.to(device)
    alpha = alpha.to(device)
    alpha_freq = alpha_freq.to(device)
    display_mask = display_mask.to(device)
    doc_feats = doc_feats.squeeze(1).float()
    doc_scores = doc_scorer(doc_feats).squeeze(-1)
    # get regression scores from the trained regression model
    reg_scores = torch.sigmoid(reg_model(doc_feats).squeeze(-1))
    # generate samples from PL-model
    ranking_scores, sampled_rankings = pl_sampler.sample(doc_scores, mask)
    ips, reg_denom = get_propensity(device, doc_scores, alpha, mask, pl_sampler, k, weights_per_rank, sampled_rankings, reuse_sampled_rankings=False)
    ips_ret = torch.clone(ips)
    #ips /= reg_denom
    ips1 = torch.clone(ips)
    #div_weight = np.sqrt((1-divergence_weight)/(divergence_weight*N))
    # div_weight =  np.sqrt(1/(N)) * np.sqrt(10**4/N)
    div_weight = 0.1
    # div_weight = 1/np.log10(N)
    #div_weight = ((1-divergence_weight)/(divergence_weight*N))
    #ips *= div_weight
    size = doc_scores.shape
    prob = nn.functional.softmax(doc_scores, dim=-1)
    log_scores = nn.functional.log_softmax(doc_scores, dim=-1)
    entropy_reg = -(prob * log_scores).sum(-1).mean()
    # expand rel_labels to match sampled ranking scores tensor size
    label_size = rel_labels.shape
    # PPO gradient mask. mask for positive reward
    clip_val = div_weight
    # clip_val = 1.
    ips[ips <= 1/clip_val] = 1.
    if mode == 'train':
        ips[ips > 1/clip_val] = 0
    else:
        clip_ratio = alpha/rho
        ips[ips > 1/clip_val] = clip_ratio[ips > 1/clip_val]
    # PPO gradient mask. mask for -ve reward
    ips1[ips1 >= clip_val] = 1
    if mode == 'train':
        ips1[ips1 < clip_val] = 0
    else:
        clip_ratio = alpha/rho
        ips1[ips1 < clip_val] = clip_ratio[ips1 < clip_val]
    # modify reward to adjust for divergence
    #rel_labels -= ips.unsqueeze(1)
    # modify reward for DR estimator
    dr_reward = (reg_scores * (1-alpha_freq/rho))#*display_mask
    #dr_reward = reg_scores - (reg_scores * (alpha_freq/rho))*display_mask
    rel_labels += dr_reward.unsqueeze(1)
    # seperate reward into +ve and -ve part, for PPO
    rel_labels1 = torch.clone(rel_labels)
    rel_labels1[rel_labels1 > 0 ] = 0
    rel_labels[rel_labels <=0] = 0
    rel_labels = rel_labels1 * ips1.unsqueeze(1) + rel_labels * ips.unsqueeze(1)
    rel_labels = rel_labels.expand(label_size[0], num_samples, label_size[2])
    rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
    doc_scores = doc_scores.unsqueeze(1).expand(size[0], num_samples, size[1])
    doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
    log_scores, _   = pl_sampler.log_scores(doc_scores, mask, k=k)
    #weights_per_rank = (1/(torch.arange(label_size[2])+1))**eta
    if weights_per_rank.shape[0] > label_size[2]:
        weights_per_rank = weights_per_rank[:label_size[2]]
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    obj = rel_labels * weights_per_rank
    #target = torch.ones_like(doc_scores)
    # take top-k elements
    #pdb.set_trace()
    obj = obj.sum(-1)
    cv = obj.mean(-1).reshape(-1, 1)
    # weigh by the log-scores - REINFORCE trick
    if mode == 'train':
        obj = log_scores * (obj/dcg_norm-cv)
        # obj = log_scores * (obj/dcg_norm)
        # aggregate scores
        obj = -torch.mean(torch.mean(obj, dim=-1)) #+ 1e-2 * entropy_reg
    else:
        obj = obj.mean(-1)
    return obj, ips_ret


def dm_obj_cv(N, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w, reg_model, mode='train', risk=False):
    weights_per_rank = torch.tensor(alpha_w + beta_w)
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank[k:] = 0.0
    entropy_loss = nn.CrossEntropyLoss()
    rel_labels = rel_labels.to(device)
    doc_feats = doc_feats.to(device)
    mask = mask.to(device)
    dcg_norm = dcg_norm.to(device)
    rho = rho.to(device)
    alpha_freq = alpha_freq.to(device)
    display_mask = display_mask.to(device)
    doc_feats = doc_feats.squeeze(1).float()
    doc_scores = doc_scorer(doc_feats).squeeze(-1)
    # get regression scores from the trained regression model
    reg_scores = torch.sigmoid(reg_model(doc_feats).squeeze(-1))
    # generate samples from PL-model
    ranking_scores, sampled_rankings = pl_sampler.sample(doc_scores, mask)
    #ips *= div_weight
    size = doc_scores.shape
    prob = nn.functional.softmax(doc_scores, dim=-1)
    log_scores = nn.functional.log_softmax(doc_scores, dim=-1)
    entropy_reg = -(prob * log_scores).sum(-1).mean()
    # expand rel_labels to match sampled ranking scores tensor size
    label_size = rel_labels.shape
    # PPO gradient mask. mask for positive reward
    # modify reward to adjust for divergence
    #rel_labels -= ips.unsqueeze(1)
    # modify reward for DR estimator
    dr_reward = reg_scores 
    #dr_reward = reg_scores - (reg_scores * (alpha_freq/rho))*display_mask
    rel_labels = dr_reward.unsqueeze(1)
    # seperate reward into +ve and -ve part, for PPO
    #rel_labels1 = torch.clone(rel_labels)
    #rel_labels1[rel_labels1 > 0 ] = 0
    #rel_labels[rel_labels <=0] = 0
    #rel_labels = rel_labels1 * ips1.unsqueeze(1) + rel_labels * ips.unsqueeze(1)
    rel_labels = rel_labels.expand(label_size[0], num_samples, label_size[2])
    rel_labels = torch.gather(rel_labels, 2, sampled_rankings)
    doc_scores = doc_scores.unsqueeze(1).expand(size[0], num_samples, size[1])
    doc_scores = torch.gather(doc_scores, 2, sampled_rankings)
    log_scores, _   = pl_sampler.log_scores(doc_scores, mask, k=k)
    #weights_per_rank = (1/(torch.arange(label_size[2])+1))**eta
    if weights_per_rank.shape[0] > label_size[2]:
        weights_per_rank = weights_per_rank[:label_size[2]]
    # LTR metric/objective func - DCG with current weights_per_rank
    # TO-DO: Make it configurable, user should be able to specify the metric
    obj = rel_labels * weights_per_rank
    #target = torch.ones_like(doc_scores)
    # take top-k elements
    #pdb.set_trace()
    obj = obj.sum(-1)
    cv = obj.mean(-1).reshape(-1, 1)
    # weigh by the log-scores - REINFORCE trick
    if mode == 'train':
        obj = log_scores * (obj/dcg_norm-cv)
        # obj = log_scores * (obj/dcg_norm)
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


def risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode):
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
    #weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    weights_per_rank = (1/(torch.arange(label_size[2])+1))
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank.requires_grad = False
    prop_counter = torch.zeros(batch_size, max_cand_size)#.to(device)
    prop_counter.requires_grad = False
    ix = torch.repeat_interleave(torch.arange(batch_size), num_rankings)
    for i in range(k):
        iy = sampled_rankings[:, :, i].reshape(batch_size * num_rankings)
        #w = torch.zeros_like(iy)
        #w[:] = torch.tensor(1/torch.log2(torch.tensor(i+2)))
        prop_counter.index_put_((ix, iy), torch.tensor(1/(i+1)).float(), accumulate=True)
    prop_counter /= num_rankings
    prop_counter = prop_counter.to(device)
    #reg_denom = torch.sqrt(torch.mean((torch.square(prop_counter.detach()/alpha.detach()) * alpha.detach()).sum(-1)))
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
    #grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
    #grad_weight = grad_weight * log_scores
    #reg = torch.mean(((ips_weight * weights_per_rank).sum(-1) * log_scores).sum(-1))
    if mode == 'train':
        cv = grad_weight.mean(-1).reshape(-1, 1)     
        obj = log_scores * (grad_weight - cv)   
        #reg = -torch.mean((grad_weight * log_scores).sum(-1)) #- 0.01 * entropy_loss(doc_scores, target)
        #reg = torch.mean(grad_weight.sum(-1)) #-  entropy_loss(doc_scores, target)
        reg =  torch.mean(obj.mean(-1))#/reg_denom #-  entropy_loss(doc_scores, target)
    else:
        reg = grad_weight.mean(-1).mean(-1)
    return reg



def get_propensity(device, doc_scores, alpha, mask, pl_sampler, k, rank_weights, sampled_rankings=None, reuse_sampled_rankings=False):
    if not reuse_sampled_rankings:
        _, sampled_rankings = pl_sampler.sample(doc_scores, mask)
    size = doc_scores.shape
    batch_size = doc_scores.shape[0]
    alpha = alpha.to(device)
    num_rankings = sampled_rankings.shape[1]
    max_cand_size = sampled_rankings.shape[2]
    prop_counter = torch.zeros(batch_size, max_cand_size)
    prop_counter.requires_grad = False
    ix = torch.repeat_interleave(torch.arange(batch_size), num_rankings)
    for i in range(k):
        iy = sampled_rankings[:, :, i].reshape(batch_size * num_rankings)
        prop_counter.index_put_((ix, iy), (rank_weights[i]).float(), accumulate=True)
    prop_counter /= num_rankings
    prop_counter = prop_counter.to(device)
    ips_ratio = prop_counter/alpha 
    reg_denom = torch.sqrt(torch.mean((torch.square(prop_counter/alpha) * alpha).sum(-1)))
    return ips_ratio, reg_denom


def risk_obj1(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode):
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
    #weights_per_rank = (1/torch.log2(torch.arange(label_size[2])+2))
    weights_per_rank = (1/(torch.arange(label_size[2])+1))**eta
    weights_per_rank = weights_per_rank.to(device)
    weights_per_rank.requires_grad = False
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
    ips = prop_counter/alpha
    #sqrt_div = torch.sqrt(((prop_counter * prop_counter)/alpha).sum(-1).mean())
    reg_denom = torch.sqrt(torch.mean((torch.square(prop_counter/alpha) * alpha).sum(-1)))
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
    grad_weight = (ips_weight * weights_per_rank).sum(-1)/reg_denom
    #out, _ = torch.max(grad_weight, dim=-1)
    grad_weight = grad_weight/grad_weight.sum(-1).reshape(-1, 1)
    #grad_weight = grad_weight/ips.mean(-1).reshape(-1, 1)
    #grad_weight = grad_weight * log_scores
    #reg = torch.mean(((ips_weight * weights_per_rank).sum(-1) * log_scores).sum(-1))
    if mode == 'train':
        cv = grad_weight.mean(-1).reshape(-1, 1)     
        obj = log_scores * (grad_weight - cv )   
        #reg = -torch.mean((grad_weight * log_scores).sum(-1)) #- 0.01 * entropy_loss(doc_scores, target)
        #reg = torch.mean(grad_weight.sum(-1)) #-  entropy_loss(doc_scores, target)
        reg =  torch.mean(obj.mean(-1)) #-  entropy_loss(doc_scores, target)
    else:
        reg = grad_weight.mean(-1).mean(-1)
    return reg#/sqrt_div


def trainer(num_queries, num_samples, k, lr, optimizer, alpha_w, beta_w,\
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
        beta_tensor = torch.tensor(beta_w).to(device)
        alpha_tensor = torch.tensor(alpha_w).to(device)
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
                display_mask = batch['display_mask']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, display_mask, mode='train')
                wandb.log({
                        "utility loss": obj.item()})
                optimizer.zero_grad()
                obj.backward()
                optimizer.step()
            doc_scorer.eval()
            # check for early stopping
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                display_mask = batch['display_mask']
                obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, display_mask, mode='test')
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
            # display_mask = torch.ones_like(rel_labels)
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, None, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        ips_file.write(str(np.mean(np.hstack(test_dcg))))



def trainer_risk(num_queries, num_samples, k, lr, optimizer, alpha_w, beta_w,\
             meta_dir, device, train_dataloader, val_dataloader, train_dataloader_risk, test_dataloader, wandb,\
                  risk_file, eta, epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        doc_scorer = DocScorer(**MLP_args)
        doc_scorer = doc_scorer.to(device)
        doc_scorer1 = DocScorer(**MLP_args) # get new instance
        doc_scorer1.load_state_dict(doc_scorer.state_dict()) # copy state
        k = k
        lr = lr
        reg_weight = 0.
        util_weight = 1.
        optimizer = optimizer
        #torch.autograd.set_detect_anomaly(True)
        beta_tensor = torch.tensor(beta_w).to(device)
        alpha_tensor = torch.tensor(alpha_w).to(device)
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
        doc_scorer = doc_scorer.to(device)
        #optimizer1 = optim.Adam(doc_scorer.parameters(), lr=0.00001, weight_decay=1e-3)
        best_score = None
        counter = 0
        patience = 3
        alternate_freq = 1
        risk_steps = 3
        for epoch in range(epochs):
            val_dcg = []
            # for g in optimizer.param_groups:
            #     g['lr'] = g['lr'] * (10/np.sqrt(num_queries))
            doc_scorer.train()
            # for g in optimizer.param_groups:
            #     g['lr'] = 1e-3
            for x, batch in enumerate(train_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                optimizer.zero_grad()
                obj =  ips_risk_obj_cv(num_queries, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_w, beta_w, mode='train')
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                wandb.log({
                        "utility loss": obj.item()})
                obj.backward()
                optimizer.step()
                
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, mode='test')
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
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write( str(np.mean(np.hstack(test_dcg))))
        
        

def trainer_dr_risk(num_queries, num_samples, k, lr, optimizer, alpha_w, beta_w,\
             meta_dir, device, train_dataloader, val_dataloader, train_dataloader_risk, test_dataloader, wandb,\
                  risk_file, eta, reg_model, epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        # get the 'trained' regression model
        for param in reg_model.parameters():
            param.requires_grad = False
        doc_scorer = DocScorer(**MLP_args)
        doc_scorer = doc_scorer.to(device)
        doc_scorer1 = DocScorer(**MLP_args) # get new instance
        doc_scorer1.load_state_dict(doc_scorer.state_dict()) # copy state
        k = k
        lr = lr
        reg_weight = 0.
        util_weight = 1.
        optimizer = optimizer
        #torch.autograd.set_detect_anomaly(True)
        beta_tensor = torch.tensor(beta_w).to(device)
        alpha_tensor = torch.tensor(alpha_w).to(device)
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
        doc_scorer = doc_scorer.to(device)
        #optimizer1 = optim.Adam(doc_scorer.parameters(), lr=0.00001, weight_decay=1e-3)
        best_score = None
        counter = 0
        patience = 3
        alternate_freq = 1
        risk_steps = 3
        for epoch in range(epochs):
            val_dcg = []
            # for g in optimizer.param_groups:
            #     g['lr'] = g['lr'] * (10/np.sqrt(num_queries))
            doc_scorer.train()
            # for g in optimizer.param_groups:
            #     g['lr'] = 1e-3
            for x, batch in enumerate(train_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                rho = batch['qid_rho']
                alpha_freq = batch['alpha_freq']
                display_mask = batch['display_mask']
                optimizer.zero_grad()
                obj =  dr_risk_obj_cv(num_queries, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w,reg_model, mode='train')
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                wandb.log({
                        "utility loss": obj.item()})
                obj.backward()
                optimizer.step()
                
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                rho = batch['qid_rho']
                alpha_freq = batch['alpha_freq']
                display_mask = batch['display_mask']
                obj= dr_risk_obj_cv(num_queries, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w,reg_model, mode='test')
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
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, display_mask=None, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write( str(np.mean(np.hstack(test_dcg))))
        
        
        
def trainer_dr(num_queries, num_samples, k, lr, optimizer, alpha_w, beta_w,\
             meta_dir, device, train_dataloader, val_dataloader, train_dataloader_risk, test_dataloader, wandb,\
                  risk_file, eta, reg_model, epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        # get the 'trained' regression model
        for param in reg_model.parameters():
            param.requires_grad = False
        doc_scorer = DocScorer(**MLP_args)
        doc_scorer = doc_scorer.to(device)
        doc_scorer1 = DocScorer(**MLP_args) # get new instance
        doc_scorer1.load_state_dict(doc_scorer.state_dict()) # copy state
        k = k
        lr = lr
        reg_weight = 0.
        util_weight = 1.
        optimizer = optimizer
        #torch.autograd.set_detect_anomaly(True)
        beta_tensor = torch.tensor(beta_w).to(device)
        alpha_tensor = torch.tensor(alpha_w).to(device)
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
        doc_scorer = doc_scorer.to(device)
        #optimizer1 = optim.Adam(doc_scorer.parameters(), lr=0.00001, weight_decay=1e-3)
        best_score = None
        counter = 0
        patience = 3
        alternate_freq = 1
        risk_steps = 3
        for epoch in range(epochs):
            val_dcg = []
            # for g in optimizer.param_groups:
            #     g['lr'] = g['lr'] * (10/np.sqrt(num_queries))
            doc_scorer.train()
            # for g in optimizer.param_groups:
            #     g['lr'] = 1e-3
            for x, batch in enumerate(train_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                rho = batch['qid_rho']
                alpha_freq = batch['alpha_freq']
                display_mask = batch['display_mask']
                optimizer.zero_grad()
                obj =  dr_obj_cv(num_queries, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w,reg_model, mode='train')
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                wandb.log({
                        "utility loss": obj.item()})
                obj.backward()
                optimizer.step()
                
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                rho = batch['qid_rho']
                alpha_freq = batch['alpha_freq']
                display_mask = batch['display_mask']
                obj = dr_obj_cv(num_queries, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w,reg_model, mode='test')
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
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, display_mask=None, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write( str(np.mean(np.hstack(test_dcg))))
        

def trainer_dr_ppo(num_queries, num_samples, k, lr, optimizer, alpha_w, beta_w,\
             meta_dir, device, train_dataloader, val_dataloader, train_dataloader_risk, test_dataloader, wandb,\
                  risk_file, eta, reg_model, epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        # get the 'trained' regression model
        for param in reg_model.parameters():
            param.requires_grad = False
        doc_scorer = DocScorer(**MLP_args)
        doc_scorer = doc_scorer.to(device)
        doc_scorer1 = DocScorer(**MLP_args) # get new instance
        doc_scorer1.load_state_dict(doc_scorer.state_dict()) # copy state
        k = k
        lr = lr
        reg_weight = 0.
        util_weight = 1.
        optimizer = optimizer
        #torch.autograd.set_detect_anomaly(True)
        beta_tensor = torch.tensor(beta_w).to(device)
        alpha_tensor = torch.tensor(alpha_w).to(device)
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
        doc_scorer = doc_scorer.to(device)
        #optimizer1 = optim.Adam(doc_scorer.parameters(), lr=0.00001, weight_decay=1e-3)
        best_score = None
        counter = 0
        patience = 3
        alternate_freq = 1
        risk_steps = 3
        for epoch in range(epochs):
            val_dcg = []
            # for g in optimizer.param_groups:
            #     g['lr'] = g['lr'] * (10/np.sqrt(num_queries))
            doc_scorer.train()
            # for g in optimizer.param_groups:
            #     g['lr'] = 1e-3
            for x, batch in enumerate(train_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                rho = batch['qid_rho']
                alpha_freq = batch['alpha_freq']
                display_mask = batch['display_mask']
                optimizer.zero_grad()
                obj, ips =  dr_ppo_obj_cv(num_queries, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w,reg_model, mode='train')
                ips_max = ips[ips>0].max()
                ips_min = ips[ips>0].min()
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                wandb.log({
                        "utility loss": obj.item()})
                wandb.log({'ips-max': ips_max.item()})
                wandb.log({'ips-min': ips_min.item()})
                obj.backward()
                optimizer.step()
                
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                rho = batch['qid_rho']
                alpha_freq = batch['alpha_freq']
                display_mask = batch['display_mask']
                obj, _ = dr_ppo_obj_cv(num_queries, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w,reg_model, mode='test')
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
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, display_mask=None, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write( str(np.mean(np.hstack(test_dcg))))
        
        


def trainer_dm(num_queries, num_samples, k, lr, optimizer, alpha_w, beta_w,\
             meta_dir, device, train_dataloader, val_dataloader, train_dataloader_risk, test_dataloader, wandb,\
                  risk_file, eta, reg_model, epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        # get the 'trained' regression model
        for param in reg_model.parameters():
            param.requires_grad = False
        doc_scorer = DocScorer(**MLP_args)
        doc_scorer = doc_scorer.to(device)
        doc_scorer1 = DocScorer(**MLP_args) # get new instance
        doc_scorer1.load_state_dict(doc_scorer.state_dict()) # copy state
        k = k
        lr = lr
        reg_weight = 0.
        util_weight = 1.
        optimizer = optimizer
        #torch.autograd.set_detect_anomaly(True)
        beta_tensor = torch.tensor(beta_w).to(device)
        alpha_tensor = torch.tensor(alpha_w).to(device)
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
        doc_scorer = doc_scorer.to(device)
        #optimizer1 = optim.Adam(doc_scorer.parameters(), lr=0.00001, weight_decay=1e-3)
        best_score = None
        counter = 0
        patience = 3
        alternate_freq = 1
        risk_steps = 3
        for epoch in range(epochs):
            val_dcg = []
            # for g in optimizer.param_groups:
            #     g['lr'] = g['lr'] * (10/np.sqrt(num_queries))
            doc_scorer.train()
            # for g in optimizer.param_groups:
            #     g['lr'] = 1e-3
            for x, batch in enumerate(train_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                rho = batch['qid_rho']
                alpha_freq = batch['alpha_freq']
                display_mask = batch['display_mask']
                optimizer.zero_grad()
                obj =  dm_obj_cv(num_queries, device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha, alpha_freq, rho, display_mask, alpha_w, beta_w,reg_model, mode='train')
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                wandb.log({
                        "utility loss": obj.item()})
                obj.backward()
                optimizer.step()
                
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, display_mask=None, mode='test')
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
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, display_mask=None, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write( str(np.mean(np.hstack(test_dcg))))



def trainer_risk1(num_queries, num_samples, k, lr, optimizer, alpha, beta,\
             meta_dir, device, train_dataloader, val_dataloader, train_dataloader_risk, test_dataloader, wandb,\
                  risk_file, eta, epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        doc_scorer = DocScorer(**MLP_args)
        doc_scorer = doc_scorer.to(device)
        doc_scorer1 = DocScorer(**MLP_args) # get new instance
        doc_scorer1.load_state_dict(doc_scorer.state_dict()) # copy state
        
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
        optimizer1 = optim.Adam(doc_scorer.parameters(), lr=lr, weight_decay=0.)
        scheduler = StepLR(optimizer1, step_size=1, gamma=0.1)
        #optimizer1.load_state_dict(optimizer1.state_dict()) # copy state
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
                optimizer.zero_grad()
                obj =   ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                wandb.log({
                        "utility loss": obj.item()})
                obj.backward()
                optimizer.step()
                #for j in range(10):
                if (x + 1)%alternate_freq == 0:
                    for j, batch in enumerate(train_dataloader_risk):
                        #torch.autograd.set_detect_anomaly(True)
                        if j >= risk_steps:
                            break 
                        rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                        alpha = batch['alpha']
                        #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                        #obj =  (10/np.log2(num_queries))* risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                        optimizer1.zero_grad()
                        #obj = (10/np.power(np.sqrt(num_queries), 2.5)) *  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                        if num_queries > 5000:
                            obj = np.power(0.9, np.sqrt(num_queries)) *  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                            #obj = (1/num_queries) *  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                        else:
                            obj = np.power(0.9, np.sqrt(num_queries)) *  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                            #obj = (1/num_queries) *  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                        wandb.log({
                            "risk loss": obj.item()})
                        #obj =  0.0
                        #obj =  (0.) *risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                        #obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                        #reg = (1/num_queries) * obj 
                        #reg = (10/np.sqrt(num_queries)) *obj 
                        total_norm = 0    
                        #print(obj)
                        #print(log_scores.sum())
                        #print(obj)
                        #optimizer.zero_grad()
                        obj.backward()
                        optimizer1.step()
                        parameters = [p for p in doc_scorer.parameters() if p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        wandb.log({
                            "grad norm": total_norm})
                        # aggregate scores
                        # obj = -util_weight * torch.mean(torch.sum(obj, dim=-1)) + reg_weight * reg
                        # optimizer.zero_grad()
                        # obj.backward()
                        # optimizer.step()
                        # if j > risk_steps:
                        #     break 
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='test')
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
            obj = ips_obj(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write( str(np.mean(np.hstack(test_dcg))))



def trainer_risk_linear(num_queries, num_samples, k, lr, optimizer, alpha, beta,\
             meta_dir, device, train_dataloader, val_dataloader, test_dataloader, wandb,\
                  risk_file, delta,eta, epochs=10, get_exposure=False,\
                      **MLP_args):
        num_samples = num_samples
        test_dcg = []
        doc_scorer = DocScorer(**MLP_args)
        doc_scorer = doc_scorer.to(device)
        doc_scorer1 = DocScorer(**MLP_args) # get new instance
        doc_scorer1.load_state_dict(doc_scorer.state_dict()) # copy state
        
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
        optimizer1 = optim.Adam(doc_scorer.parameters(), lr=lr, weight_decay=0.)
        scheduler = StepLR(optimizer1, step_size=1, gamma=0.1)
        #optimizer1.load_state_dict(optimizer1.state_dict()) # copy state
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
                optimizer.zero_grad()
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='train')
                wandb.log({
                    "ips objective": obj.item()})
                #batch_risk = batch
                #rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                reg = delta * np.sqrt(1/num_queries) * risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
                #reg = np.power(0.9, np.sqrt(num_queries)) * risk_obj1(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                wandb.log({
                    "risk objective": reg.item()})
                obj += reg
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                obj.backward()
                optimizer.step()
                #for j in range(10):

            # for x, batch in enumerate(train_dataloader):
            #     rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
            #     optimizer.zero_grad()
            #     obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
            #     wandb.log({
            #         "ips objective": obj.item()})
            #     alpha = batch['alpha']
            #     reg = np.sqrt(10/num_queries) * risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
            #     wandb.log({
            #         "risk objective": reg.item()})
            #     obj += reg
            #     #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
            #     obj.backward()
            #     optimizer.step()
            #     #for j in range(10):
                
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='test')
                #rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                reg = -delta  * risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='test')
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
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='test')
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
        doc_scorer1 = DocScorer(**MLP_args) # get new instance
        doc_scorer1.load_state_dict(doc_scorer.state_dict()) # copy state
        
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
        optimizer1 = optim.Adam(doc_scorer.parameters(), lr=lr, weight_decay=0.)
        scheduler = StepLR(optimizer1, step_size=1, gamma=0.1)
        #optimizer1.load_state_dict(optimizer1.state_dict()) # copy state
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
                optimizer.zero_grad()
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='train')
                wandb.log({
                    "ips objective": obj.item()})
                #batch_risk = batch
                #rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                reg = delta * np.sqrt(1/num_queries) * risk_obj1(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='train')
                #reg = np.power(0.9, np.sqrt(num_queries)) * risk_obj1(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                wandb.log({
                    "risk objective": reg.item()})
                obj += reg
                #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
                obj.backward()
                optimizer.step()
                #for j in range(10):

            # for x, batch in enumerate(train_dataloader):
            #     rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
            #     optimizer.zero_grad()
            #     obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, mode='train')
            #     wandb.log({
            #         "ips objective": obj.item()})
            #     alpha = batch['alpha']
            #     reg = np.sqrt(10/num_queries) * risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
            #     wandb.log({
            #         "risk objective": reg.item()})
            #     obj += reg
            #     #obj =  risk_obj(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k)
            #     obj.backward()
            #     optimizer.step()
            #     #for j in range(10):
                
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='test')
                #rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                reg = -delta * np.sqrt(1/num_queries)  * risk_obj1(device, rel_labels, doc_feats, alpha, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='test')
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
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5": np.mean(np.hstack(test_dcg))})
        risk_file.write( str(np.mean(np.hstack(test_dcg))))


def trainer_regression(num_queries, num_samples, k, lr, optimizer, alpha_w, beta_w,\
             meta_dir, device, train_dataloader, val_dataloader, train_dataloader_risk, test_dataloader, wandb,\
                  risk_file, eta, epochs=10, get_exposure=False,\
                      **MLP_args):
        EPS = 1e-11
        num_samples = num_samples
        test_dcg = []
        doc_scorer = DocScorer(**MLP_args)
        doc_scorer = doc_scorer.to(device)
        doc_scorer1 = DocScorer(**MLP_args) # get new instance
        doc_scorer1.load_state_dict(doc_scorer.state_dict()) # copy state
        k = k
        lr = lr
        reg_weight = 0.
        util_weight = 1.
        optimizer = optimizer
        #torch.autograd.set_detect_anomaly(True)
        beta_tensor = torch.tensor(beta_w).to(device)
        alpha_tensor = torch.tensor(alpha_w).to(device)
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
        doc_scorer = doc_scorer.to(device)
        #optimizer1 = optim.Adam(doc_scorer.parameters(), lr=0.00001, weight_decay=1e-3)
        best_score = None
        counter = 0
        patience = 3
        alternate_freq = 1
        risk_steps = 5
        for epoch in range(int(epochs * 1)):
            val_dcg = []
            # for g in optimizer.param_groups:
            #     g['lr'] = g['lr'] * (10/np.sqrt(num_queries))
            doc_scorer.train()
            # for g in optimizer.param_groups:
            #     g['lr'] = 1e-3
            for x, batch in enumerate(train_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels_reg'], batch['feats'], batch['mask'], batch['dcg_norm']
                alpha = batch['alpha']
                #beta = batch['qid_rho1']
                display_mask = batch['display_mask']
                alpha_freq = batch['alpha_freq']
                beta_freq = batch['beta_freq']
                # expected alpha
                rho = batch['qid_rho']
                #rel_labels += (beta - beta_freq)/rho
                optimizer.zero_grad()
                # Cross-entropy objective
                rel_labels = rel_labels.to(device)
                doc_feats = doc_feats.to(device)
                mask = mask.to(device)
                rho = rho.to(device)
                display_mask = display_mask.to(device)
                dcg_norm = dcg_norm.to(device)
                alpha = alpha.to(device)
                alpha_freq = alpha_freq.to(device)
                batch_size = alpha.shape[0]
                doc_feats = doc_feats.squeeze(1).float()
                #doc_feats = doc_feats.float()
                rel_labels = rel_labels.squeeze(1)
                #rel_labels = rel_labels
                doc_scores = doc_scorer(doc_feats).squeeze(-1)
                #doc_scores = doc_scorer(doc_feats)
                doc_scores = torch.sigmoid(doc_scores)
                obj = rel_labels * torch.log(doc_scores.clip(min=EPS)) + ( alpha_freq/rho - rel_labels) * torch.log((1 - doc_scores).clip(min=EPS))
                #obj = rel_labels * torch.log(doc_scores.clip(min=EPS)) + ( 1 - rel_labels) * torch.log((1 - doc_scores).clip(min=EPS))
                #obj *= display_mask
                #obj *= mask
                obj = -obj.sum(-1)
                wandb.log({
                        "regression model loss": obj.item()})
                obj.backward()
                optimizer.step()
                
            #scheduler.step()
            doc_scorer.eval()
            for i, batch in enumerate(val_dataloader):
                rel_labels, doc_feats, mask, dcg_norm = batch['labels_reg'], batch['feats'], batch['mask'], batch['dcg_norm']
                display_mask = batch['display_mask']
                display_mask = display_mask.to(device)
                alpha_freq = batch['alpha_freq']
                beta_freq = batch['beta_freq']
                #beta = batch['qid_rho1']
                # expected alpha
                rho = batch['qid_rho']
                #rel_labels += (beta - beta_freq)/rho
                rho = rho.to(device)
                alpha_freq = alpha_freq.to(device)
                rel_labels = rel_labels.to(device)
                doc_feats = doc_feats.to(device)
                mask = mask.to(device)
                dcg_norm = dcg_norm.to(device)
                alpha = alpha.to(device)
                batch_size = alpha.shape[0]
                doc_feats = doc_feats.squeeze(1).float()
                #doc_feats = doc_feats.float()
                rel_labels = rel_labels.squeeze(1) 
                #rel_labels = rel_labels
                doc_scores = doc_scorer(doc_feats).squeeze(-1)
                #doc_scores = doc_scorer(doc_feats)
                doc_scores = torch.sigmoid(doc_scores)
                obj = rel_labels * torch.log(doc_scores.clip(min=EPS)) + ( alpha_freq/rho - rel_labels) * torch.log((1 - doc_scores).clip(min=EPS))
                #obj *= display_mask
                #obj *= mask
                obj = obj.sum(-1)
                val_dcg.append(obj.detach().to('cpu').numpy())
            final_val_dcg = np.mean(np.hstack(val_dcg))
            wandb.log({
                        "val-ce": final_val_dcg})
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
                        "epoch-reg": epoch})
                        break

        doc_scorer.eval()
        for i, batch in enumerate(test_dataloader):
            rel_labels, doc_feats, mask, dcg_norm = batch['labels'], batch['feats'], batch['mask'], batch['dcg_norm']
            obj = ips_obj_cv(device, rel_labels, doc_feats, mask, dcg_norm, doc_scorer, pl_sampler, num_samples, k, eta, alpha_w, beta_w, mode='test')
            test_dcg.append(obj.to('cpu').numpy())
        wandb.log({
        "test nDCG@5-reg": np.mean(np.hstack(test_dcg))})
        #risk_file.write( str(np.mean(np.hstack(test_dcg))))
        return doc_scorer





