'''
main file for training counterfactual PL model on click data with trust-bias. 
Assuming single logging policy trained on a fraction of a relevance judgement data.
'''

import argparse
from xmlrpc.client import Boolean
import yaml
import warnings
import os
import logging

from sklearn import datasets
import os, json, pdb
import pandas as pd
import numpy as np

import torch.nn as nn
from src.models.nnmodel import DocScorer
from src.data.data_loader_click import ClickLogDataloader
from src.data.data_loader_direct import ClickLogDataloader as ClickLogDataloaderDirect
from src.data.data_loader import LTRDataLoader
from src.models.PLRanker import PlackettLuceModel
from torch.utils.data import DataLoader
from src.utils.dataloader import MultiEpochsDataLoader
from src.models.PLRankerClick import PLRanker
from src.utils.click_model import get_alpha, trust_bias, trust_bias_misspec
import wandb
from src.utils.click_trainer import trainer, trainer_risk, trainer_dr_ppo, trainer_regression

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', type=str,
                 required=True)
parser.add_argument('--num_sessions', type=int,  
                required=True)
parser.add_argument('--risk', type=int,  
                required=True) 
parser.add_argument('--dataset', type=str,  
                required=True)
parser.add_argument('--noise', type=float,  
                required=True)
parser.add_argument('--T', type=float, required=True)
parser.add_argument('--run', type=int, required=False )
args = parser.parse_args()

os.environ["WANDB_SILENT"] = "true"

wandb.login()
if args.risk == 1:
    wandb.init(project=args.dataset + str(args.noise) + "_risk_T=%0.2f_PBM=1.0_exp"%args.T, entity="shashankg7")
else:
    wandb.init(project=args.dataset + str(args.noise) + "_ips_T=%0.2f_PBM=1.0_exp"%args.T, entity="shashankg7")
wandb.run.name = args.dataset + str(args.noise) + "_sessions=" + str(args.num_sessions)
wandb.run.save()



def main():
    config = yaml.safe_load(open('./config/config.yaml', 'rb'))
    simulation_config = yaml.safe_load(open('./config/clickmodel_config.yaml', 'rb'))
    hp_dict_ips = json.load(open('./results/hp_dict_ips.json', 'r'))
    hp_dict_risk = json.load(open('./results/hp_dict_risk.json', 'r'))
    hp_dict_risk = hp_dict_risk[args.dataset]
    hp_dict_ips = hp_dict_ips[args.dataset]
    eta = simulation_config['clickmodel']['eta']
    grid = np.log10(np.array(list((hp_dict_risk.keys())), dtype=np.int64))
    grid_ips = np.log10(np.array(list((hp_dict_ips.keys())), dtype=np.int64))
    pow_10 = np.floor(np.log10(args.num_sessions))
    grid_idx = np.argmin(np.abs(grid - pow_10))
    grid_point = str(int(np.power(10, grid[grid_idx])))
    lr_risk = hp_dict_risk[grid_point]
    grid_idx = np.argmin(np.abs(grid_ips - pow_10))
    grid_point = str(int(np.power(10, grid_ips[grid_idx])))
    lr_ips = hp_dict_ips[grid_point]
    print(config)
    print(simulation_config)
    alpha = simulation_config['bias_params']['alpha']
    beta = simulation_config['bias_params']['beta']
    root_dir = config['dataset']['root_dir']
    root_dir_pred = config['dataset']['predict_dir']
    dataset_dir = os.path.join(root_dir, args.dataset)
    predict_dir = os.path.join(os.path.join(root_dir_pred, args.dataset), 'click_logs')
    predict_dir_val = os.path.join(os.path.join(root_dir_pred, args.dataset), 'click_logs_val')
    predict_dir = predict_dir + str('_%d'%(args.num_sessions))
    predict_dir_val = predict_dir_val + str('_%d'%(args.num_sessions))
    meta_dir = os.path.join(os.path.join(root_dir_pred, args.dataset), 'click_metadata')
    meta_dir = meta_dir + str('_%d'%(args.num_sessions))
    logging_dir = os.path.join(os.path.join(root_dir_pred, args.dataset), 'logging_policy')
    click_model_type = simulation_config['clickmodel']['type']
    device = 'cuda:0'
    dataset_fold = config['dataset']['fold']
    query_normalize = config['dataset']['normalize']
    batch_size = config['hyperparams']['batch_size']
    if args.dataset == 'MQ2007':
        doc_feat_size = 46
    elif args.dataset == 'MSLR30K':
        doc_feat_size = 136
    elif args.dataset == 'Yahoo':
        doc_feat_size = 699
    elif args.dataset == 'ISTELLA':
        doc_feat_size = 220
    else:
        raise ValueError('UNKNOWN DATASET')
    k = config['ranking']['k']
    max_cand_size = config['ranking']['max_cand_size']
    if click_model_type == 'pbm':
        alpha, alpha_exp = get_alpha(eta, max_cand_size, k)
        beta1 = np.zeros_like(alpha)
        beta1[:k] = alpha
        beta = beta1
    else:
        alpha, beta = trust_bias(max_cand_size, k)
        # try with misspecified alpha, beta
        # alpha, beta = trust_bias_misspec(max_cand_size, k)
        alpha_exp = alpha
    # if args.dataset == 'MQ2007' and int(args.num_sessions) < 10000 :
    #     lr = 0.001
    # elif args.dataset == 'MQ2007' and int(args.num_sessions) >= 10000 :
    #     lr = 0.0001
    # elif args.dataset == 'MSLR30K' and int(args.num_sessions) < 10000 :
    #     lr = 0.001
    # elif args.dataset == 'MSLR30K' and int(args.num_sessions) >= 10000 :
    #     lr = 0.0001
    # elif args.dataset == 'Yahoo' and int(args.num_sessions) < 10000 :
    #     lr = 0.001
    # elif args.dataset == 'Yahoo' and int(args.num_sessions) >= 10000 :
    #     lr = 0.0001
    # elif args.dataset == 'ISTELLA' and int(args.num_sessions) < 10000 :
    #     lr = 0.001
    # elif args.dataset == 'ISTELLA' and int(args.num_sessions) >= 10000 :
    #     lr = 0.0001
    # else:
    #     raise ValueError('UNKNOWN DATASET')
    lr = config['hyperparams']['lr']
    print(lr)
    batch_size = config['hyperparams']['batch_size']
    optimizer = config['hyperparams']['optimizer']
    dataset_path = os.path.join(dataset_dir, dataset_fold)
    train_svmlight = os.path.join(dataset_path, 'train.txt')
    val_svmlight = os.path.join(dataset_path, 'vali.txt')
    test_svmlight = os.path.join(dataset_path, 'test.txt')
    # prepare dataloaders for train/test/val
    clipping_val = 10/np.sqrt(args.num_sessions)
    train_ltr_dataloader = ClickLogDataloader(meta_dir=meta_dir, click_dir=predict_dir, mode='train', feat_vec_dim=doc_feat_size, max_cand_size=max_cand_size, clip=clipping_val, click_model=click_model_type, estimator='dr')# clip_alpha=clipping_val)
    #train_ltr_dataloader1 = ClickLogDataloaderDirect(meta_dir=meta_dir, click_dir=predict_dir, mode='train', feat_vec_dim=doc_feat_size, max_cand_size=max_cand_size, clip=clipping_val, click_model=click_model_type)#, clip_alpha=clipping_val)
    train_dataloader = MultiEpochsDataLoader(train_ltr_dataloader, batch_size=batch_size,
                        shuffle=True, num_workers=2, persistent_workers=True)
    train_dataloader1 = MultiEpochsDataLoader(train_ltr_dataloader, batch_size=1,
                        shuffle=True, num_workers=8, persistent_workers=True)
    val_ltr_dataloader = ClickLogDataloader(meta_dir=meta_dir, click_dir=predict_dir, mode='val', feat_vec_dim=doc_feat_size, max_cand_size=max_cand_size, clip=0., click_model=click_model_type, estimator='dr')
    #val_ltr_dataloader1 = ClickLogDataloaderDirect(meta_dir=meta_dir, click_dir=predict_dir, mode='val', feat_vec_dim=doc_feat_size, max_cand_size=max_cand_size, clip=0., click_model=click_model_type)
    val_dataloader = MultiEpochsDataLoader(val_ltr_dataloader, batch_size=batch_size,
                        shuffle=False, num_workers=2, persistent_workers=True)
    val_dataloader1 = MultiEpochsDataLoader(val_ltr_dataloader, batch_size=batch_size,
                        shuffle=False, num_workers=8, persistent_workers=True)
    test_ltr_dataloader = LTRDataLoader(qrel_path=test_svmlight, train=False, scaler=None, k=k, noise=float(args.noise), max_cand_size=max_cand_size, meta_dir=meta_dir, mode='test', save=True)
    rel_scale = 0.25
    test_ltr_dataloader.set_label(rel_scale)
    test_dataloader = MultiEpochsDataLoader(test_ltr_dataloader, batch_size=batch_size,
                            shuffle=False, num_workers=2, persistent_workers=True)
    
    if args.risk == 1:
        results_file_risk = open(os.path.join('./results/test_results', args.dataset + '_' + str(args.T) + '_' + str(args.num_sessions) + '_' + str(args.job_id) + '_' + 'ppo_exp'), 'w')
        reg_model = trainer_regression(num_queries=args.num_sessions, num_samples=config['ranking']['num_samples'], k=k, \
                            lr=lr_risk, optimizer=optimizer, alpha_w=alpha, beta_w=beta, \
                                train_qid_map=None, num_docs=max_cand_size,\
                            meta_dir = meta_dir,  device=device, \
                                wandb=wandb, risk_file=results_file_risk,
                                train_dataloader=train_dataloader1,
                                val_dataloader=val_dataloader1,
                                train_dataloader_risk=train_dataloader1,
                                 test_dataloader=test_dataloader,\
                                    eta=eta,
                                 **{'doc_feat_dim':doc_feat_size,\
                             'hidden_dim1':config['docscorer']['hidden_dim1'],\
                             'hidden_dim2':config['docscorer']['hidden_dim2']})
        
        logging_model = trainer_dr_ppo(num_queries=args.num_sessions, num_samples=config['ranking']['num_samples'], k=k, \
                            lr=lr_risk, optimizer=optimizer, alpha_w=alpha, beta_w=beta, \
                                train_qid_map=None, num_docs=max_cand_size,\
                            meta_dir = meta_dir,  device=device, \
                                wandb=wandb, risk_file=results_file_risk,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                train_dataloader_risk=train_dataloader,
                                 test_dataloader=test_dataloader,\
                                    eta=eta, reg_model=reg_model,
                                 **{'doc_feat_dim':doc_feat_size,\
                             'hidden_dim1':config['docscorer']['hidden_dim1'],\
                             'hidden_dim2':config['docscorer']['hidden_dim2']})
    else:
        results_file_ips = open(os.path.join('./results/test_results', args.dataset + '_' + str(args.T) + '_' + str(args.num_sessions) + '_' + str(args.job_id) + '_'  + 'ips_exp'), 'w')
        logging_model = trainer(num_queries=args.num_sessions, num_samples=config['ranking']['num_samples'], k=k, \
                            lr=lr_ips, optimizer=optimizer, alpha_w=alpha, beta_w=beta, \
                                train_qid_map=None, num_docs=max_cand_size,\
                            meta_dir = meta_dir,  device=device, \
                                wandb=wandb, ips_file=results_file_ips,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                 test_dataloader=test_dataloader,\
                                    eta=eta,
                                 **{'doc_feat_dim':doc_feat_size,\
                             'hidden_dim1':config['docscorer']['hidden_dim1'],\
                             'hidden_dim2':config['docscorer']['hidden_dim2']})

if __name__ == '__main__':
    main()
