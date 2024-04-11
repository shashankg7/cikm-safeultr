'''
main file for training counterfactual PL model on click data with trust-bias. 
Assuming single logging policy trained on a fraction of a relevance judgement data.
'''

import argparse
import yaml
import warnings
import os, copy
import logging
import pickle
import torch

from sklearn import datasets
from sklearn.model_selection import train_test_split
import os, json, pdb
import pandas as pd
import numpy as np

import pytorch_lightning as pl
import torch.nn as nn
from src.models.nnmodel import DocScorer
from src.data.data_loader import LTRDataLoader
from src.data.data_loader_logger import LTRLoggerDataLoader
from src.data.data_loader_simul import LTRClickDataLoader
from src.models.PLRanker import PlackettLuceModel
from torch.utils.data import DataLoader
#from src.utils.dataloader import MultiEpochsDataLoader
from src.utils.click_model import get_alpha, trust_bias
from src.models.PLRanker import PLRanker
from src.utils.click_simulation import click_simulation
from src.models.nnmodel import DocScorer, Logger

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



parser = argparse.ArgumentParser()
parser.add_argument('--job_id', type=str,
                 required=True)
parser.add_argument('--num_sessions', type=int,  
                required=True)
parser.add_argument('--dataset', type=str,  
                required=True)
parser.add_argument('--noise', type=float,  
                required=True)
parser.add_argument('--run', type=int, required=False)
parser.add_argument('--T', type=float, required=True)
parser.add_argument('--deterministic', type=str, required=True)
parser.add_argument('--fraction', type=float, required=True)
args, unknown = parser.parse_known_args()

#os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"]="offline"


wandb.login()
wandb.init(project=(args.dataset + str(args.noise)), entity="shashankg7")
wandb.run.name = args.dataset + str(args.noise) + "Logging_sessions" + str(args.num_sessions)
wandb.run.save()


def main():
    config = yaml.safe_load(open('./config/config.yaml', 'rb'))
    simulation_config = yaml.safe_load(open('./config/clickmodel_config.yaml', 'rb'))
    if args.deterministic == 'True':
        deterministic = True
    else:
        deterministic = False
    print(config)
    print(simulation_config)
    eta = simulation_config['clickmodel']['eta']
    click_model_type = simulation_config['clickmodel']['type']
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
        alpha_exp = alpha
        beta_exp = beta
    #alpha1 = np.zeros(max_cand_size)
    #alpha1[:k] = alpha
    #alpha = alpha1
    #beta1 = np.zeros_like(alpha)
    #beta1[:k] = alpha
    #beta = beta1
    num_sessions = config['ranking']['num_sessions']
    # override num_sessions with input from argument - REMOVE FROM CONFIG FILE LATER
    num_sessions = args.num_sessions
    num_sessions_val = config['ranking']['num_sessions_val']
    lr = config['hyperparams']['lr']
    batch_size = config['hyperparams']['batch_size']
    optimizer = config['hyperparams']['optimizer']
    num_samples = config['ranking']['num_samples']
    dataset_path = os.path.join(dataset_dir, dataset_fold)
    train_svmlight = os.path.join(dataset_path, 'train.txt')
    val_svmlight = os.path.join(dataset_path, 'vali.txt')
    test_svmlight = os.path.join(dataset_path, 'test.txt')
    # dataloader for training logging policy. 
    # Rel_scale = 1/4 * rel(q,d) for training logging_policy
    train_ltr_dataloader = LTRDataLoader(qrel_path=train_svmlight, train=True, normalize=query_normalize, k=k, noise=float(args.noise), max_cand_size=max_cand_size, meta_dir=meta_dir, mode='train', save=True)
    #rel_scale = 0.25
    rel_scale = 0.25
    train_df = copy.deepcopy(train_ltr_dataloader.df)
    train_ltr_dataloader.set_label(rel_scale)
    train_ltr_dataloader.max_list_size = max_cand_size
    train_dataloader = DataLoader(train_ltr_dataloader, batch_size=batch_size,
                        shuffle=True, num_workers=4, persistent_workers=True)
    # get the standard scaler from the train object.
    # TO-DO: Move to log1p scaling, independent of the training data batch. 
    if query_normalize:
        scaler = train_ltr_dataloader.scaler
    else:
        scaler = None
    val_ltr_dataloader = LTRDataLoader(qrel_path=val_svmlight, train=False, normalize=query_normalize, scaler=scaler, k=k, noise=float(args.noise), max_cand_size=max_cand_size, meta_dir=meta_dir, mode='val', save=True)
    if args.dataset == 'MQ2007':
        rel_scale = 0.5
        # rel_scale = 0.025
    else:
        rel_scale = 0.25
        #rel_scale = 0.02
    val_ltr_dataloader.set_label(rel_scale)
    #val_ltr_dataloader.max_list_size = max_cand_size
    train_qid_map = train_ltr_dataloader.qid_map
    #train_qid_map = {**train_qid_map, **val_ltr_dataloader.qid_map}
    #train_ltr_dataloader.qid_map = train_qid_map
    with open(os.path.join(meta_dir, 'train_qid_map'), 'wb') as fp:
        pickle.dump(train_qid_map, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #val_ltr_dataloader = LTRDataLoader(qrel_path=val_svmlight, train=False, normalize=query_normalize, scaler=scaler, k=k, max_cand_size=max_cand_size, meta_dir=meta_dir, mode='val', save=True)
    val_dataloader = DataLoader(val_ltr_dataloader, batch_size=batch_size,
                            shuffle=False, num_workers=4, persistent_workers=True)

    test_ltr_dataloader = LTRDataLoader(qrel_path=test_svmlight, train=False, normalize=query_normalize, scaler=scaler, k=k, noise=float(args.noise), max_cand_size=max_cand_size, meta_dir=meta_dir, mode='test', save=True)
    #test_ltr_dataloader.max_list_size = max_cand_size
    if args.dataset == 'MQ2007':
        rel_scale = 0.5
        # rel_scale = 0.025
    else:
        rel_scale = 0.25
    test_ltr_dataloader.set_label(rel_scale)
    test_dataloader = DataLoader(test_ltr_dataloader, batch_size=batch_size,
                            shuffle=False, num_workers=4, persistent_workers=True)
    
    #train_df = copy.deepcopy(train_ltr_dataloader.df)
    val_df = val_ltr_dataloader.df
    # Dataloader for logging policy
    train_df_logging = copy.deepcopy(train_df)
    train_df_logging['label'] = train_df_logging['label'] * rel_scale
    logging_ltr_dataloader = LTRLoggerDataLoader(train_df_logging, k=k, train_frac=args.fraction)
    logging_dataloader = DataLoader(logging_ltr_dataloader, batch_size=batch_size,
                        shuffle=True, num_workers=4, persistent_workers=True)
    
    # Dataloader for (train) click simulation
    # change rel_label to (1-noise)/40
    if args.dataset == 'MQ2007':
        rel_scale = 0.5
        # rel_scale = 0.025
    else:
        rel_scale = 0.25
        #rel_scale = 0.082
        #rel_scale = 0.0825
    #train_ltr_dataloader.set_label(rel_scale)
    train_df_click = copy.copy(train_df)
    train_df_click['label'] = train_df_click['label'] * rel_scale
    #train_df = train_ltr_dataloader.df
    click_sim_dl = LTRClickDataLoader(num_sessions=num_sessions, train_df=train_df_click, k=k)
    click_sim_dataloader = DataLoader(click_sim_dl, batch_size=batch_size * 2,
                        shuffle=False, num_workers=4, persistent_workers=True)
    
    click_sim_dl.max_list_size = max_cand_size
    # Dataloader for (val.) click simulation
    click_sim_dl_val = LTRClickDataLoader(num_sessions=num_sessions_val, train_df=val_df, k=k)
    #val_ltr_dataloader.set_label(rel_scale)
    click_sim_dataloader_val = DataLoader(click_sim_dl_val, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    click_sim_dl_val.max_list_size = max_cand_size
    # wandb logger 
    wandb_logger = WandbLogger(project=args.dataset)


    trainer = pl.Trainer(
        gpus=1,
        logger=wandb_logger, 
        detect_anomaly=True,
        callbacks=[EarlyStopping(monitor="val-dcg@5", mode="max")],
        max_epochs=10
    )
                                #  callbacks=[prediction_writer])
    if os.path.isfile(os.path.join(logging_dir, 'logging_policy')):
        logging_model = PLRanker.load_from_checkpoint(os.path.join(logging_dir, 'logging_policy'),\
                                                     out_dir=predict_dir,\
                                                        out_dir_val=predict_dir_val,\
                                                            meta_dir=meta_dir, T=args.T, alpha=alpha, alpha_exp=alpha_exp, click_model_type = click_model_type,\
                                                                  noise=float(args.noise), beta=beta)
        logging_model.T = args.T
        logging_model.click_model_type = click_model_type
        logging_model.alpha = alpha
        logging_model.beta = beta
        logging_model.alpha_exp = alpha_exp 
        logging_model.beta_exp = beta_exp
    else:
        logging_model = PLRanker(num_samples=config['ranking']['num_samples'], k=k, \
                            lr=lr, optimizer=optimizer, alpha=alpha, alpha_exp=alpha_exp, beta=beta, \
                            meta_dir = meta_dir, train_qid_map=train_qid_map,\
                                 out_dir=predict_dir, out_dir_val=predict_dir_val,
                                  num_docs=max_cand_size, noise=float(args.noise), deterministic=deterministic, T=args.T, click_model_type=click_model_type,\
                                      **{'doc_feat_dim':doc_feat_size,\
                             'hidden_dim1':config['docscorer']['hidden_dim1'],\
                             'hidden_dim2':config['docscorer']['hidden_dim2']})
        trainer.fit(logging_model, logging_dataloader, val_dataloader)
        trainer.save_checkpoint(os.path.join(logging_dir, 'logging_policy'))
    trainer.test(logging_model, test_dataloader)
    # run click simulation
    trainer.predict(logging_model, click_sim_dataloader)
    logging_model.qid_clicks = logging_model.qid_clicks.to('cpu')
    if not deterministic:
        #doc_prob_rank = (logging_model.doc_per_rank_prob/((logging_model.query_freq).unsqueeze(1).unsqueeze(1))).nan_to_num_().detach().numpy()
        doc_prob_rank = logging_model.doc_per_rank_prob.nan_to_num_().detach().numpy()
        # doc_per_rank_prob_det has the total alpha[k] per q,d pair. After normalizing it, don't need to compute exp_alpha
        #doc_prob_rank_freq = (logging_model.doc_per_rank_prob_det/((logging_model.query_freq).unsqueeze(1).unsqueeze(1))).nan_to_num_().detach().numpy()
        doc_prob_rank_freq = (logging_model.doc_per_rank_prob_det.sum(-1)/((logging_model.query_freq).unsqueeze(1))).nan_to_num_().detach().numpy()
        exp_beta_freq = (logging_model.doc_per_rank_prob_beta.sum(-1)/((logging_model.query_freq).unsqueeze(1))).nan_to_num_().detach().numpy()
        exp_alpha = (doc_prob_rank * alpha[:k].reshape(1,1,k)).sum(-1)
        exp_beta = (doc_prob_rank * beta[:k].reshape(1,1,k)).sum(-1)
        exp_alpha_beta = (doc_prob_rank * (beta[:k].reshape(1,1,k) + alpha[:k].reshape(1,1,k))).sum(-1)
        # exp_alpha_freq = (doc_prob_rank_freq * alpha[:k].reshape(1,1,k)).sum(-1) 
        # exp_beta_freq = (doc_prob_rank_freq * beta[:k].reshape(1,1,k)).sum(-1)
        exp_alpha_freq = doc_prob_rank_freq
        #exp_beta_freq = doc_prob_rank_freq
        display_mask = logging_model.doc_per_rank_prob_det.sum(-1).to(bool).to(torch.int)
    else:
        #doc_prob_rank = (logging_model.doc_per_rank_prob_det/((logging_model.query_freq).unsqueeze(1)).nan_to_num_().detach().numpy()
        doc_prob_rank = logging_model.doc_per_rank_prob_det.detach().numpy()
        #doc_prob_rank = logging_model.doc_per_rank_prob_det.detach().numpy()
        exp_alpha = doc_prob_rank
    qid_ctr_map = (logging_model.qid_clicks.sum(-1)/((logging_model.query_freq).unsqueeze(1))).nan_to_num_().detach().numpy()
    #print(qid_ctr_map.sum())
    #exp_beta  = (doc_prob_rank * np.array(beta[:k]).reshape(1,1,k)).sum(-1)
    #exp_beta  = doc_prob_rank
    # click simulation callback
    # logging_model.get_exposure = True
    # trainer.predict(logging_model, exp_dataloader)
    np.save(os.path.join(meta_dir, 'doc_rank_prob'), doc_prob_rank)
    np.save(os.path.join(meta_dir, 'alpha'), exp_alpha)
    np.save(os.path.join(meta_dir, 'beta'), exp_beta)
    np.save(os.path.join(meta_dir, 'beta_freq'), exp_beta_freq)
    np.save(os.path.join(meta_dir, 'alpha_freq'), exp_alpha_freq)
    np.save(os.path.join(meta_dir, 'alpha_beta'), exp_alpha_beta)
    np.save(os.path.join(meta_dir, 'ctr_map'), qid_ctr_map)
    np.save(os.path.join(meta_dir, 'display_mask'), display_mask)
    # split query set into train/val split
    active_queries = np.where(logging_model.query_freq > 0)[0]
    query_train, query_val = train_test_split(active_queries, test_size=0.15, random_state=42)
    np.save(os.path.join(meta_dir, 'query_train'), query_train)
    np.save(os.path.join(meta_dir, 'query_val'), query_val)
    #logging_model.click_sim_val = True
    # run (val) click simulation
    #trainer.predict(logging_model, click_sim_dataloader_val)
    #click_simulation(trainer, logging_policy=logging_model, click_dl=click_sim_dataloader, alpha=alpha, beta=beta, k=k)
    


if __name__ == '__main__':
    main()
