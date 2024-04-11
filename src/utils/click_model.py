import numpy as np
import copy


def get_alpha(eta, max_cand_size, k):
    exposure = (1/(np.arange(max_cand_size)+1))**eta
    exposure_prop = copy.deepcopy(exposure)
    exposure[k:] = 0
    return exposure, exposure_prop

def get_alpha_beta(eta, max_cand_size, k):
    theta = (1/(np.arange(max_cand_size)+1))**eta
    theta[k:] = 0
    eta1_neg = 0.65
    eta_neg = eta1_neg * 1/np.minimum(np.arange(max_cand_size)+1, 10)
    eta_pos = 1 - (np.minimum(np.arange(max_cand_size)+1, 20) + 1)/100 
    beta = theta * eta_neg
    alpha = theta * eta_pos - beta
    return alpha, beta

def trust_bias(max_cand_size, k):
    ranks = np.arange(max_cand_size)
    pos_bias = 0.35*1/(ranks+1.) + 0.65/(1.+0.05*ranks)
    eplus = 1./(1.+0.005*ranks)
    emin = 0.65/(ranks+1.)
    alpha = pos_bias*(eplus - emin) 
    beta = pos_bias*emin
    alpha[k:] = 0.
    beta[k:] = 0.
    return alpha, beta

def trust_bias_misspec(max_cand_size, k, error=0.00):
    ranks = np.arange(max_cand_size)
    pos_bias = 0.35*1/(ranks+1.) + 0.65/(1.+0.05*ranks)
    eplus = 1./(1.+0.005*ranks)
    emin = 0.65/(ranks+1.)
    alpha = pos_bias*(eplus - emin) 
    beta = pos_bias*emin
    # add error in the alpha, beta
    alpha = error * alpha + (1-error) * np.mean(alpha[:k])
    beta = error * beta + (1 - error) * np.mean(beta[:k])
    alpha[k:] = 0.
    beta[k:] = 0.
    return alpha, beta

if __name__ == '__main__':
    print(get_alpha_beta(2, 120, 5))