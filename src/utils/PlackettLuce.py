
import os, json, pdb
import pandas as pd
import numpy as np
import torch


class PlackettLuceModel:
    '''
    Plackett-Luce distribution
    '''
    
    def __init__(self, n_samples):
        '''
        Inputs:
            n_samples: #sampled rankings to generate per query
        '''
        self.n_samples = n_samples
        self.eps = 1e-08
    
    def __reverse_logcumsumexp(self, ranking_scores, mask, k):
        '''
        Custom implementation of logcumsumexp operation on ranking_scores with masking. 
        Returns document placement probability per rank, and the log-scores(for REINFORCE training)
        '''
        # need to sum from position-i to last position in the ranking. Cumsum gives score from pos-0 till pos-i, hence using flip
        log_norm = torch.flip(torch.logcumsumexp(torch.flip(ranking_scores, [2]), -1), [2])
        doc_prob_per_rank = torch.cumprod(normalized_scores, dim=-1)
        # get-log-score from top-k elements
        log_score = torch.sum(ranking_scores[:, :, :k], dim=-1) - torch.sum(log_norm[:, :, :k], dim=-1)
        return log_score, doc_prob_per_rank

    
    def prob_per_rank(self, ranking_scores):
        '''
        Get placement prob. of document per rank.
        '''
        log_norm = torch.flip(torch.logcumsumexp(torch.flip(ranking_scores, [2]), -1), [2])
        normalized_scores = torch.exp(ranking_scores - log_norm)
        doc_prob_per_rank = normalized_scores
        return doc_prob_per_rank

        
    def sample(self, logits, mask, T=1):
        '''
        Input:
            logits: shape= #queriesInBatch * max_rank_length (with scores per document)
            Generate sampled rankings for the mini-batch of queries with logit scores per doc/item.
            mask: Mask per query. For each query, indicates the padding used 
        '''
        # Sampling via Gumbel distribution. 
        # Step1: Sample i.i.d. gumbel noise
        # Step2: Add the gumbel noise to the scores
        # Sort based on the noisy scores, take top-K. Results in a sampling from the PL model
        # T = temperature. T < 1 makes the sampling more deterministic, T > 1 makes it more uniform. 
        logits = logits/T
        self.size = logits.shape
        logits = logits.unsqueeze(1).expand(self.size[0], self.n_samples, self.size[1])
        unif = torch.rand_like(logits)
        gumbel_scores = logits - torch.log(-torch.log(unif + self.eps ) )
        gumbel_scores = gumbel_scores * mask.unsqueeze(1)
        ranking_scores, sampled_rankings = torch.sort(gumbel_scores, descending=True, dim=-1, stable=True)
        return (ranking_scores, sampled_rankings)

    def log_scores(self, ranking_scores, mask, k):
        '''
        Output: 
            computes log-score given samples    
            output size: #queriesInBatch
        '''
        mask1 = mask.unsqueeze(1).expand(mask.shape[0], self.n_samples, mask.shape[1])
        log_score, doc_prob_per_rank = self.__reverse_logcumsumexp(ranking_scores, mask1, k)
        return log_score, doc_prob_per_rank        
