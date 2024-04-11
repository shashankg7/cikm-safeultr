import json, os
import numpy as np
import pandas as pd

hp_dir = './results/hp'
hp_dict_ips = {}
hp_dict_risk = {}
hp_files = os.listdir(hp_dir)

datasets = []
methods = []
lrs = []
num_sessions = []
val_dcg = []

for hp_file in hp_files:
    hp_val = open(os.path.join(hp_dir, hp_file)).read()
    val_dcg.append(float(hp_val))
    dataset, num_session, lr, method = hp_file.strip().split('_')
    datasets.append(dataset)
    num_sessions.append(int(num_session))
    lrs.append(float(lr))
    methods.append(method)

val_df = pd.DataFrame(np.column_stack([datasets, methods, num_sessions, lrs, val_dcg]), 
                               columns=['dataset', 'method', 'num_sessions', 'lr', 'val_dcg'])

val_df['num_sessions'] = pd.to_numeric(val_df['num_sessions'])
val_df['lr'] = pd.to_numeric(val_df['lr'])
val_df['val_dcg'] = pd.to_numeric(val_df['val_dcg'])

for dataset in list(set(datasets)):
    val_df_dataset = val_df[val_df['dataset'] == dataset]
    val_df_dataset_risk = val_df_dataset[val_df_dataset['method'] == 'risk'].reset_index(drop=True)
    val_df_dataset_ips = val_df_dataset[val_df_dataset['method'] == 'ips'].reset_index(drop=True)
    if dataset not in hp_dict_risk:
        hp_dict_risk[dataset] = {}
    if dataset not in hp_dict_ips:
        hp_dict_ips[dataset] = {}
    for num_session in list(set(num_sessions)):
        val_df_dataset_risk_session = val_df_dataset_risk[val_df_dataset_risk['num_sessions'] == num_session].reset_index(drop=True)
        val_df_dataset_risk_session = val_df_dataset_risk_session.sort_values('val_dcg', ascending=False).reset_index(drop=True)
        dcg_sorted = val_df_dataset_risk_session['val_dcg'].tolist()
        lrs = val_df_dataset_risk_session['lr'].tolist()
        if len(set(dcg_sorted)) == 1:
            hp_dict_risk[dataset][num_session] = sorted(lrs)[-1]
            #hp_dict_ips[dataset][num_session] = sorted(lrs)[-1]
        else:
            dcg_index = dcg_sorted.index(max(dcg_sorted))
            hp_dict_risk[dataset][num_session] = val_df_dataset_risk_session['lr'].loc[dcg_index]
        
        val_df_dataset_ips_session = val_df_dataset_ips[val_df_dataset_ips['num_sessions'] == num_session].reset_index(drop=True)
        val_df_dataset_ips_session = val_df_dataset_ips_session.sort_values('val_dcg', ascending=False).reset_index(drop=True)
        dcg_sorted = val_df_dataset_ips_session['val_dcg'].tolist()
        lrs = val_df_dataset_ips_session['lr'].tolist()
        if len(set(dcg_sorted)) == 1:
            #hp_dict_risk[dataset][num_session] = sorted(lrs)[0]
            hp_dict_ips[dataset][num_session] = sorted(lrs)[0]
        else:
            dcg_index =  dcg_sorted.index(max(dcg_sorted))
            hp_dict_ips[dataset][num_session] = val_df_dataset_ips_session['lr'].loc[dcg_index]


with open('./results/hp_dict_ips.json', 'w') as fp:
    json.dump(hp_dict_ips, fp)

with open('./results/hp_dict_risk.json', 'w') as fp:
    json.dump(hp_dict_risk, fp)

print(hp_dict_ips)
print(hp_dict_risk)

