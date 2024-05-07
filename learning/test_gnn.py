#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:38:45 2021

@author: abdel
"""


import os
import sys
import torch
import torch_geometric
from pathlib import Path
from model import GNNPolicy
from data_type import GraphDataset
from utils import process

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p','--problem',
        help='MILP instance type to process.',
        default = 'facilities',
    )
    parser.add_argument(
        '-n','--num',
        help='nums of train set',
        default = '1000',
    )
    parser.add_argument(
        '-d','--partition',
        help='data_partition',
        default = 'test',
    )
    args = parser.parse_args()

    problem = args.problem
    nums_instance = args.num
    data_partition = args.partition
    n_sample = -1
    normalize = True
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    batch_test  = 256
    
    loss_fn = torch.nn.BCELoss()
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-normalize':
            normalize = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
            
    test_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data/{problem}/{data_partition}")).glob("*.pt") ][:int(0.2*n_sample if n_sample != -1 else -1)]
    
    test_data = GraphDataset(test_files)
    
    
# TO DO : learn something from the data
    test_loader = torch_geometric.loader.DataLoader(test_data, 
                                                     batch_size=batch_test, 
                                                     shuffle=False, 
                                                     follow_batch=['constraint_features_s',
                                                                   'constraint_features_t',
                                                                   'variable_features_s',
                                                                   'variable_features_t'])
    
    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(f'../checkpoint/policy_{problem}_{nums_instance}.pkl'))

    test_loss, test_acc = process(policy, 
                                    test_loader, 
                                    loss_fn, 
                                    device,
                                    optimizer=None,
                                    normalize=normalize)
    with open('output.txt','a') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(f"policy_{problem}_gnn_{nums_instance} on {data_partition} for {len(test_files)} behaviours" )   
        print(f"Test loss: {test_loss:0.3f}, accuracy {test_acc:0.3f}" )
        print("")
        
    sys.stdout = original_stdout
    print(f"policy_{problem}_gnn_{nums_instance} outputed" )