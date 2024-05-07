# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:23:29 2022

@author: aglabassi
"""

import os
import sys
import numpy as np
from pathlib import Path
from joblib import dump, load

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
            
    # return np.array(X),np.array(y), np.array(depths)
    return np.array(X), np.array(y), np.array(depths)
        

if __name__ == '__main__':
    
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
        default = '1',
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
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-n_sample':
            n_sample = int(sys.argv[i + 1])
        
        if problem == 'facilities':
            if nums_instance == '1':
                f = lambda x: x[18] - x[38] + 0.2
            if nums_instance == '1000':
                f = lambda x: -x[25] - x[28] + x[5] 
        elif problem == 'setcover':
            if nums_instance == '1':
                f = lambda x: -x[27]*x[28]+x[8]
            if nums_instance == '1000':
                f = lambda x: -x[15] - x[25] + x[5] + x[8] -0.5
        elif problem == 'FCMCNF':
            if nums_instance == '1':
                f = lambda x: -x[25] - x[33] + x[5]
            if nums_instance == '1000':
                f = lambda x: x[18] - x[28] - x[38]
        else: raise NotImplementedError 
    
    test_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data_svm/{problem}/{data_partition}")).glob("*.csv") ][:int(0.2*n_sample if n_sample != -1 else -1)]
    # print(train_files)
    X_test, y_test, depths_test = get_data(test_files)
    
    y_pred = np.apply_along_axis(f, 1, X_test)
    # y_pred = python_execute(traversal, X_test)
    # test_acc = model.score(X_test,y_test, np.min(depths_test, axis=1))
    y_pred[y_pred >= 0] = 1.0
    y_pred[y_pred < 0] = -1.0
    print(y_pred[:5])
    print(y_test[:5])
    test_acc = (y_pred == y_test).mean()
    
    with open('output.txt','a') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(f"policy_{problem}_symb_{nums_instance} on {data_partition} for {len(test_files)} behaviours" )   
        print(f"Accuracy : {test_acc}")
        print("")
    
    sys.stdout = original_stdout
    print(f"policy_{problem}_symb_{nums_instance} outputed" )
        
        
        
        
        
        
        
    
        
