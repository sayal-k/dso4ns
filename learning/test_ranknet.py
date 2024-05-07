# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch_geometric
from pathlib import Path
from model1 import RankNet
from data_type import GraphDataset
from utils import process, process_ranknet
import numpy as np

def get_data(files):
    
    X = []
    y = []
    depths = []
    
    for file in files:
        
        f_array = np.loadtxt(file)
        features = f_array[:-1]
        comp_res = f_array[-1]
        if features.shape[0]==40:
            X.append(features)
            y.append(comp_res)
            depths.append(np.array([f_array[18], f_array[-3]]))
        
        
    
    return np.array(X, dtype=np.float64),np.array(y), np.array(depths)
    #return np.array(X, dtype=np.float32), np.array(y, dtype=np.long), np.array(depths)


    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p','--problem',
        help='MILP instance type to process.',
        default = 'WPMS',
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
    batch_test  = 1
    
    loss_fn = torch.nn.BCELoss()
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problem':
            early_stopping = int(sys.argv[i + 1])
        if sys.argv[i] == '-normalize':
            normalize = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
            
    test_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data_svm/{problem}/{data_partition}")).glob("*.csv") ][:int(0.2*n_sample if n_sample != -1 else -1)]

    X_test, y_test, _ = get_data(test_files)
    

    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test).unsqueeze(1)
        
    X_test.to(device)
    y_test.to(device)    

    policy = RankNet().to(device)
    policy.load_state_dict(torch.load(f'../checkpoint/policy_{problem}_ranknet_{nums_instance}.pkl'))
    
    test_loss, test_acc = process_ranknet(policy, 
                                    X_test, y_test, 
                                    loss_fn, 
                                    device,
                                    optimizer=None)
        
    with open('output.txt','a') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(f"policy_{problem}_ranknet_{nums_instance} on {data_partition} for {len(test_files)} behaviours" )
        print(f"Test loss: {test_loss:0.3f}, accuracy {test_acc:0.3f}" )
        print("")    
    
    sys.stdout = original_stdout
    print(f"policy_{problem}_ranknet_{nums_instance} outputed")
