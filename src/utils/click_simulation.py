'''
Generates clicks given predictions from the logging policy
'''
import torch
import numpy as np
import argparse

def click_simulation(datasets, max_clicks, num_points):
    upper_limit_exp = int(np.log10(max_clicks))
    lower_limit_exp = 2
    points = np.around(np.logspace(lower_limit_exp, upper_limit_exp, num_points, endpoint=True))
    #points = points[::2]
    params_file = open('./params.txt', "w")
    #datasets = ['MSLR30K', 'Yahoo', 'ISTELLA']
    #datasets = ['MQ2007']
    datasets = eval(datasets)
    noises = [0.0, 0.1]
    for dataset in datasets:
        for point in points:
            txt = '--dataset ' + dataset + ' ' +  '--num_sessions ' + str(int(point)) + '\n'
            params_file.write(txt)
    params_file.close()
    return points

def ips_training(datasets, points, num_runs):
    clicks = []
    #datasets = ['MQ2007']
    #datasets = ['MSLR30K', 'Yahoo', 'ISTELLA']
    datasets = eval(datasets)
    ips_param_file = open('./params_ips.txt', 'w')
    noises = [0.0, 0.1]
    
    for dataset in datasets:
        for click in points:
            for run in range(num_runs):
                txt = '--dataset ' + dataset + ' ' +'--num_sessions ' + str(int(click)) + '\n'
                ips_param_file.write(txt)
        
    ips_param_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper_limit', type=np.int64,
                    required=True)
    parser.add_argument('--num_sessions', type=int,  
                    required=True)
    parser.add_argument('--datasets', type=str, required=True )
    parser.add_argument('--repeatations', type=int, required=True)
    args, unknown = parser.parse_known_args()
    points = click_simulation(datasets=args.datasets, max_clicks=args.upper_limit, num_points=args.num_sessions)
    ips_training(datasets=args.datasets, points=points, num_runs=args.repeatations)
