#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:43:54 2021

@author: abdel
"""

def load_src(name, fpath):
     import os, imp
     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("data_type", "../learning/data_type.py" )
load_src("model", "../learning/model.py" )

import torch
import sys
sys.path.append("../learning")
import time
import numpy as np
from pyscipopt import Nodesel
from data_type import BipartiteGraphPairData
from model import GNNPolicy, RankNet
from line_profiler import LineProfiler
from joblib import dump, load



class CustomNodeSelector(Nodesel):

    def __init__(self, sel_policy='', comp_policy=''):
        self.sel_policy = sel_policy
        self.comp_policy = comp_policy
        self.sel_counter = 0
        self.comp_counter = 0

        
    def nodeselect(self):
        
        self.sel_counter += 1
        policy = self.sel_policy
        
        if policy == 'estimate':
            res = self.estimate_nodeselect()
        elif policy == 'dfs':
            res = self.dfs_nodeselect()
        elif policy == 'breadthfirst':
            res = self.breadthfirst_nodeselect()
        elif policy == 'bfs':
            res = self.bfs_nodeselect()
        elif policy == 'random':
            res = self.random_nodeselect()
        else:
            res = {"selnode": self.model.getBestNode()}
            
        return res
    
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        policy = self.comp_policy
        
        if policy == 'estimate':
            res = self.estimate_nodecomp(node1, node2)
        elif policy == 'dfs':
            res = self.dfs_nodecomp(node1, node2)
        elif policy == 'breadthfirst':
            res = self.breadthfirst_nodecomp(node1, node2)
        elif policy == 'bfs':
            res = self.bfs_nodecomp(node1, node2)
        elif policy == 'random':
            res = self.random_nodecomp(node1, node2)
        else:
            res = 0
            
        return res
    
    #BFS
    def bfs_nodeselect(self):
        return {'selnode':self.model.getBfsSelNode() }
        
        
        
    #Estimate 
    def estimate_nodeselect(self):
        return {'selnode':self.model.getEstimateSelNode() }
    
    def estimate_nodecomp(self, node1,node2):
        
        #estimate 
        estimate1 = node1.getEstimate()
        estimate2 = node2.getEstimate()
        if (self.model.isInfinity(estimate1) and self.model.isInfinity(estimate2)) or \
            (self.model.isInfinity(-estimate1) and self.model.isInfinity(-estimate2)) or \
            self.model.isEQ(estimate1, estimate2):
                lb1 = node1.getLowerbound()
                lb2 = node2.getLowerbound()
                
                if self.model.isLT(lb1, lb2):
                    return -1
                elif self.model.isGT(lb1, lb2):
                    return 1
                else:
                    ntype1 = node1.getType()
                    ntype2 = node2.getType()
                    CHILD, SIBLING = 3,2
                    
                    if (ntype1 == CHILD and ntype2 != CHILD) or (ntype1 == SIBLING and ntype2 != SIBLING):
                        return -1
                    elif (ntype1 != CHILD and ntype2 == CHILD) or (ntype1 != SIBLING and ntype2 == SIBLING):
                        return 1
                    else:
                        return -self.dfs_nodecomp(node1, node2)
     
        
        elif self.model.isLT(estimate1, estimate2):
            return -1
        else:
            return 1
        
        
        
    # Depth first search        
    def dfs_nodeselect(self):
        
        selnode = self.model.getPrioChild()  #aka best child of current node
        if selnode == None:
            
            selnode = self.model.getPrioSibling() #if current node is a leaf, get 
            # a sibling
            if selnode == None: #if no sibling, just get a leaf
                selnode = self.model.getBestLeaf()
                
        return {"selnode": selnode}
    
    def dfs_nodecomp(self, node1, node2):
        return -node1.getDepth() + node2.getDepth()
    
    
    
    # Breath first search
    def breadthfirst_nodeselect(self):
        
        selnode = self.model.getPrioSibling()
        if selnode == None: #no siblings to be visited (all have been LP-solved), since breath first, 
        #we take the heuristic of taking the best leaves among all leaves
            
            selnode = self.model.getBestLeaf() #DOESTN INCLUDE CURENT NODE CHILD !
            if selnode == None: 
                selnode = self.model.getPrioChild()
        
        return {"selnode": selnode}
    
    def breadthfirst_nodecomp(self, node1, node2): 
        
        d1, d2 = node1.getDepth(), node2.getDepth()
        
        if d1 == d2:
            #choose the first created node
            return node1.getNumber() - node2.getNumber()
        
        #less deep node => better
        return d1 - d2
        
     
     #random
    def random_nodeselect(self):
        return {"selnode": self.model.getBestNode()}
    def random_nodecomp(self, node1,node2):
        return -1 if np.random.rand() < 0.5 else 1

    



class OracleNodeSelectorAbdel(CustomNodeSelector):

    def __init__(self, oracle_type, optsol=0, prune_policy='estimate', inv_proba=0, sel_policy=''):

        super().__init__(sel_policy=sel_policy)
        self.oracle_type = oracle_type
        self.optsol = optsol
        self.prune_policy = prune_policy 
        self.inv_proba = inv_proba
        self.sel_policy = sel_policy
        self.inf_counter  = 0
        self.inference_time = 0
        
    
    def nodecomp(self, node1, node2, return_type=False):
        
        self.comp_counter += 1
        
        if self.oracle_type == "optimal_plunger":            
        
            d1 = self.is_sol_in_domaine(self.optsol, node1)
            d2 = self.is_sol_in_domaine(self.optsol, node2)
            inv = np.random.rand() < self.inv_proba
            
            if d1 and d2:
                res, comp_type = self.dfs_nodecomp(node1, node2), 0
            elif d1:
                res = comp_type = -1
                self.inf_counter += 1
                
            
            elif d2:
                res = comp_type = 1
                self.inf_counter += 1
            
            else:
                res, comp_type = self.estimate_nodecomp(node1, node2), 10              
            
            inv_res = -1 if res == 1 else 1
            res = inv_res if inv else res
            
            return res if not return_type  else  (res, comp_type)
        else:
            raise NotImplementedError

    
    def is_sol_in_domaine(self, sol, node):
        #By partionionning, it is sufficient to only check what variable have
        #been branched and if sol is in [lb, up]_v for v a branched variable
        
        bvars, bbounds, btypes = node.getAncestorBranchings()
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
            if btype == 0:#LOWER BOUND
                if sol[bvar] < bbound:
                    return False
            else: #btype==1:#UPPER BOUND
                if sol[bvar] > bbound:
                    return False
        
        return True
            
            
    def setOptsol(self, optsol):
        self.optsol = optsol
        

class OracleNodeSelectorEstimator_RankNet(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, nums_instances, device, sel_policy='', n_primal=2):
        super().__init__(sel_policy=sel_policy)
        
        
        policy = RankNet()

        policy.load_state_dict(torch.load(f"./checkpoint/policy_{problem}_ranknet_{nums_instances}.pkl", map_location=device)) #run from main

        policy.to(device)
        
        self.policy = policy
        self.device = device
        

        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        self.inference_time = []
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
            
        start = time.perf_counter() 
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        decision =  self.policy(torch.tensor(f1, dtype=torch.float, device=self.device), torch.tensor(f2, dtype=torch.float, device=self.device))      
        self.inference_time.append(time.perf_counter() - start)
    
        self.inf_counter += 1
        
        return -1 if decision < 0.5 else 1


class OracleNodeSelectorEstimator_SVM(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, nums_instances, sel_policy='', n_primal=2):
        super().__init__(sel_policy=sel_policy)
        
        self.policy = load(f'./checkpoint/policy_{problem}_svm_{nums_instances}.pkl')
        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        self.inference_time = []
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
        start = time.perf_counter()
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        X = np.hstack((f1,f2))
        X = X[np.newaxis, :]        
        
        decision = self.policy.predict(X)[0]
        self.inference_time.append(time.perf_counter() - start)
        
        self.inf_counter += 1
        
        return -1 if decision < 0.5 else 1

    
class OracleNodeSelectorEstimator_GP(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, nums_instances, sel_policy='', n_primal=2):
        super().__init__(sel_policy=sel_policy)
        
        self.policy = load(f'./checkpoint/policy_{problem}_gp_{nums_instances}.pkl')
        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        self.inference_time = []
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
        start = time.perf_counter() 
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        X = np.hstack((f1,f2))
        X = X[np.newaxis, :]
        decision = self.policy.predict(X)[0]
        self.inference_time.append(time.perf_counter() - start)
        self.inf_counter += 1
        
        return -1 if decision < 0 else 1

class OracleNodeSelectorEstimator_Symb(CustomNodeSelector):
    '''
    It uses a symbolic regression model to estimate the node comparison.
    The symbolic regression model is trained on the features of the nodes
    and the comparison result.
    '''
    def __init__(self, problem, comp_featurizer, nums_instances, sel_policy='', n_primal=2):
        super().__init__(sel_policy=sel_policy)
        
        if problem == 'WPMS':
            if nums_instances == '1':
                f = lambda x: x[31]-(x[3]+x[8]) ** 2
            if nums_instances == '1000':
                f = lambda x: -x[25] + x[31]*x[5] + x[4] + x[8]
                #-x26 + x32*x6 + x5 + x9
        elif problem == 'facilities':
            if nums_instances == '1':
                f = lambda x: x[18] - x[38] + 0.2
            if nums_instances == '1000':
                f = lambda x: -x[25] - x[28] + x[5] 
        elif problem == 'setcover':
            if nums_instances == '1':
                f = lambda x: -x[27]*x[28]+x[8]
            if nums_instances == '1000':
                f = lambda x: -x[15] - x[25] + x[5] + x[8] -0.5
        elif problem == 'cauctions':
            if nums_instances == '1':
                f = lambda x: -x[27]*x[28]+x[8]
            if nums_instances == '1000':
                f = lambda x: -x[28]-x[4]+x[8]
        elif problem == 'indset':
            if nums_instances == '1':
                f = lambda x: x[31]-(x[3]+x[8]) ** 2
            if nums_instances == '1000':
                f = lambda x: -x[28] + 2*x[8]
        elif problem == 'GISP':
            if nums_instances == '1':
                f = lambda x: np.log(x[5]/x[25])
            if nums_instances == '1000':
                f = lambda x: x[15] - x[25] + x[5]
        elif problem == 'FCMCNF':
            if nums_instances == '1':
                f = lambda x: -x[25] - x[33] + x[5]
            if nums_instances == '1000':
                f = lambda x: x[18] - x[28] - x[38]
        else: raise NotImplementedError  
        
        self.policy = f
        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        self.inference_time = []
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
        start = time.perf_counter() 
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        X = np.hstack((f1,f2))
        decision = self.policy(X)
        self.inference_time.append(time.perf_counter() - start)
        self.inf_counter += 1
        
        return -1 if decision < 0 else 1    
    
    
class OracleNodeSelectorEstimator_Symm(CustomNodeSelector):
    '''
    It uses a symbolic regression model to estimate the node comparison.
    The symbolic regression expression is trained on the features of one node for symmetry
    and the comparison result of two nodes.
    '''
    def __init__(self, problem, comp_featurizer, nums_instances, sel_policy='', n_primal=2):
        super().__init__(sel_policy=sel_policy)
        
        if problem == 'WPMS':
            if nums_instances == '1':
                f = lambda x: (x[1] + x[3] + x[8]) ** 2
            if nums_instances == '1000':
                f = lambda x: x[13] - x[5] - x[7] + np.exp(x[6])
        elif problem == 'facilities':
            if nums_instances == '1':
                f = lambda x: x[10] - (x[18] + x[2] + x[6] + x[7]) ** 2
                # x11 - (x19 + x3 + x7 + x8)**2
            if nums_instances == '1000':
                f = lambda x: x[15]*x[8]/x[18] - x[18] - np.exp(x[7])  
        elif problem == 'setcover':
            if nums_instances == '1':
                # x20 - x8*(x19**2 + x7 - 0.5)
                f = lambda x: x[19]-x[7]*(x[18]**2 +x[6] -0.5)
            if nums_instances == '1000':
                # 0.5*x6*(x10*x11**2 - x8)
                f = lambda x: 0.5*x[5]*(x[9]*(x[10]**2)-x[7])
        elif problem == 'FCMCNF':
            if nums_instances == '1':
                # x8*(-x6 + x8 + exp(x2 + x8))
                f = lambda x: x[7]*(-x[5] + x[7] + np.exp(x[1] + x[7]))                
            if nums_instances == '1000':# x16*x9/x19 - x19 - exp(x8)
                f = lambda x: x[15]*x[8]/x[18] - x[18] - np.exp(x[7])
        elif problem == 'indset':
            if nums_instances == '1000':
                f = lambda x: -x[28] + 2*x[8]
        elif problem == 'GISP':
            if nums_instances == '1':
                # -x13**2 + x15 - 2*x19 + x4
                f = lambda x: -x[12]**2 + x[14] - 2 * x[18] + x[3]
            if nums_instances == '1000':
                # x16 + x3 + exp(x12 - x6) + 0.5
                f = lambda x: x[15] + x[2] + np.exp(x[11]-x[5])
        else: raise NotImplementedError  
        
        self.policy = f
        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        self.inference_time = []
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
        start = time.perf_counter() 
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        X = np.vstack((f1,f2))
        decision = 1 / (1 + np.exp(self.policy(X[0]) - self.policy(X[1])))
        self.inference_time.append(time.perf_counter() - start)
        self.inf_counter += 1
        
        return -1 if decision < 0.5 else 1    
        
    
    
class OracleNodeSelectorEstimator(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, device, feature_normalizor, nums_instances, n_primal=2, use_trained_gnn=True, sel_policy=''):
        super().__init__(sel_policy=sel_policy)
        
        
        
        policy = GNNPolicy()
        if use_trained_gnn: 
            # policy.load_state_dict(torch.load(f"./checkpoint/policy_{problem}.pkl", map_location=device)) #run from main
            policy.load_state_dict(torch.load(f"./checkpoint/policy_{problem}_{nums_instances}.pkl", map_location=device))
        else:
            print("Using randomly initialized gnn")
            
        policy.to(device)
        
        self.policy = policy
        self.comp_featurizer = comp_featurizer
        self.device = device
        self.feature_normalizor = feature_normalizor
        
        self.n_primal = n_primal
        self.best_primal = np.inf #minimization
        self.primal_changes = 0
        
        self.fe_time = 0
        self.fn_time = 0
        self.inference_time = []
        self.inf_counter = 0
        
        self.scores = dict()
        
        
        
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_featurizer.set_LP_feature_recorder(LP_feature_recorder)
        
        self.init_solver_cpu = LP_feature_recorder.init_time
        self.init_cpu_gpu = LP_feature_recorder.init_cpu_gpu_time
        
        self.fe_time = 0
        self.fn_time = 0
        # self.inference_time = []
        self.inf_counter = 0

        
    
    def nodecomp(self, node1,node2):
        
        self.comp_counter += 1        
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
            
            
        #begin inference process
        comp_scores = [-1,-1]
        
        for comp_idx, node in enumerate([node1, node2]):
            n_idx = node.getNumber()
        
            if n_idx in self.scores:
                comp_scores[comp_idx] = self.scores[n_idx]
            else:

                _time, g =  self.comp_featurizer.get_graph_for_inf(self.model, node)
                
                self.fe_time += _time

                
                start = time.time()
                g = self.feature_normalizor(*g)[:-1]
                self.fn_time += (time.time() - start)
                
                start = time.perf_counter()
                score = self.policy.forward_graph(*g).item()
                self.scores[n_idx] = score 
                comp_scores[comp_idx] = score
                self.inference_time.append(time.perf_counter() - start)
                
        self.inf_counter += 1
        
        return -1 if comp_scores[0] > comp_scores[1] else 1
