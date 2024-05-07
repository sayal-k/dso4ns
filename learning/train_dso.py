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
from dso import DeepSymbolicRegressor

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
    return np.array(X, dtype=np.float64), np.array(y), np.array(depths)
        

if __name__ == '__main__':
    
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
        default = '1',
    )
    args = parser.parse_args()

    problem = args.problem
    nums_instance = args.num

    n_sample = -1
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-n_sample':
            n_sample = int(sys.argv[i + 1])
        
    
    train_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data_svm/{problem}/train_{nums_instance}")).glob("*.csv") ][:n_sample]
    
    valid_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data_svm/{problem}/valid")).glob("*.csv") ][:int(0.2*n_sample if n_sample != -1 else -1)]
    # print(train_files)
    X,y,depths = get_data(train_files)
    X_valid, y_valid, depths_valid = get_data(valid_files)
    
    print(f"X shape {X.shape}")

    model = DeepSymbolicRegressor("./config.json")

    model.fit(X,y)
    
    print(model.program_.pretty())
    
    y_pred = model.predict(X_valid)
    y_pred[y_pred >= 0] = 1.0
    y_pred[y_pred < 0] = -1.0
    valid_acc = (y_pred == y_valid).mean()
    
    print(f"Accuracy on validation set : {valid_acc}")

    import datetime 
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open('output.txt','a') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(f"policy_{problem}_dso_{nums_instance}  over {len(train_files)} behaviours" )
        print(f"best expression: {model.program_.pretty()}")   
        print(f"Validation Accuracy : {valid_acc}")
        print("")
    
        
        
        
        
        
        
        
    
        
