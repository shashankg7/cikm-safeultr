'''
Generates clicks given predictions from the logging policy
'''
import torch
import numpy as np


def hp_range():
    hp_file = open('hp_file.txt', "w")
    hp_file_click = open('hp_file_click.txt', "w")
    hidden_dim1_range = [150]
    hidden_dim2_range = [75]
    lr_range = [1e-4, 1e-3]
    #datasets = ['MSLR30K', 'Yahoo', 'ISTELLA']
    datasets = ['MQ2007']
    points = [100, 1000, 10000, 100000]
    for dataset in datasets:
        for point in points:
            for hidden_dim1 in hidden_dim1_range:
                for hidden_dim2 in hidden_dim2_range:
                    for lr in lr_range:
                        text = '--dataset ' + dataset + ' ' + '--num_sessions ' + str(point) + ' ' + '--hidden_dim1 ' + str(hidden_dim1) + ' ' + '--hidden_dim2 ' + str(hidden_dim2) + ' ' + '--lr ' + str(lr) + '\n'
                        hp_file.write(text)
    hp_file.close() 

    for dataset in datasets:
        for point in points:
            text = '--dataset ' + dataset + ' ' + '--num_sessions ' + str(int(point)) + '\n'
            hp_file_click.write(text)
    hp_file_click.close() 

if __name__ == '__main__':
    hp_range()