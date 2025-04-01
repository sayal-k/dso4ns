#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:12:41 2022

@author: aglabassi
"""

import numpy as np
import pyscipopt.scip as sp
import os
import networkx as nx



def get_bipartite(n1,n2,p):
    nx.bipartite_random_graph
    
    
    


def gen_maxcut_graph_clauses(rng,n,er_prob,p=0.3):
    
    G = nx.erdos_renyi_graph(n=n, p=er_prob, seed=int(rng.get_state()[1][0]), directed=True) 
    
    divider = rng.randint(1,6)
    G = nx.algorithms.bipartite.generators.random_graph(n//divider, n - n//divider, p=er_prob, seed=int(rng.get_state()[1][0]))
    
    n_edges = len(G.edges)
    edges = list(G.edges)
    
    
    added_edges = 0
    while added_edges < n_edges*p:
        i,j = rng.randint(0,n), rng.randint(0,n)
        if (i,j) not in edges and (j,i) not in edges:
            added_edges += 1 
            edges.append((i,j))
        
    
    
    
    
    return [ ( f'v{i},v{j}', 1 )  for (i,j) in edges ] +  [ (f'-v{i},-v{j}', 1) for (i,j) in edges ]
            
    
            
        
    
    
    


#Ramon Bejar
#https://computational-sustainability.cis.cornell.edu/cpaior2013/pdfs/ansotegui.pdf
def get_clauses(rng,H,W, n_piece, n_obstacle):

    pieces = [(rng.randint(1,H+1), rng.randint(1,W+1)) for _ in range(n_piece) ] + [(1,1) ]*n_obstacle
    
    #Generate obstacles, aka filled pieces
    obstacle_pos = []
    while(len(obstacle_pos) != n_obstacle):
        candidat = (rng.randint(0,H), rng.randint(0,W))
        if candidat not in obstacle_pos:
            obstacle_pos.append(candidat)
            

        
    
    #Soft clauses
    clauses = [(f'x{k}', piece[0]*piece[1]) for k,piece in enumerate(pieces[:n_piece])]
    
    

    #Hard clauses
    #No  two times same row/colomn per piece
    for k,piece in enumerate(pieces):
        h,w = piece
        clauses.append((f'-x{k},' + ','.join([ f'r{i}_{k}' for i in range(0,H-h+1)  ]),np.inf))

        for i in range(0, H-h+1):
            for j in range(i+1, H-h+1):
                clauses.append((f'-x{k},-r{i}_{k},-r{j}_{k}',np.inf))

        
        clauses.append((f'-x{k},' + ','.join([ f'c{j}_{k}' for j in range(0,W-w+1)  ]), np.inf))

        for i in range(0, W-w+1):
            for j in range(i+1,W-w+1):
                clauses.append((f'-x{k},-c{i}_{k},-c{j}_{k}',np.inf))

    #Place obstacles
    for k,pos in enumerate(obstacle_pos):
        i,j = pos
        reajusted_k = n_piece + k
        clauses.append((f'x{reajusted_k}', np.inf))
        clauses.append((f'r{i}_{reajusted_k}', np.inf))
        clauses.append((f'c{j}_{reajusted_k}', np.inf))


    # #No overlapping between pieces(and obstacles)
    
    
    cl_s  = []
    for s,piece1 in enumerate( pieces ):
        h,w = piece1

        for k,piece2 in enumerate(pieces):
            if k != s:
    
                for i in range(0, H-h+1):
                    for j in range(0, W-w+1):
                        for l in range(i, i+h):
                            for m in range(j, j+w):
    
                                cl = (f'-r{i}_{s},-c{j}_{s},-r{l}_{k},-c{m}_{k}',np.inf)
                                cl_s.append(cl)
            

    

    return clauses + cl_s



def write_lp(clauses, filename):
    
    ''' 
        clauses (in conj normal form )  : list of clauses to be "and-ed" with their weiths
        
        Clause  : string representing a conjunctive (or's') clause, variable seperated by ',', 
        negation of variable  represented by -.
        
        
        
        Ex : 2*(A1 or not(A2)) and 1*(not(C)) == [ ('A,-A2', 2) , ('-C',1) ]
        
        '''
    
    var_names = dict() #maps var appearing in order i in clauses (whatever the name ) to y_i

    with open(filename, 'w') as file:
        file.write("maximize\nOBJ:")
        file.write("".join([f" +{clause[1]}cl_{idx}" for idx,clause in enumerate(clauses) if clause[1] < np.inf ]))

        
        
        file.write("\n\nSubject to\n")
    
        for idx,clause in enumerate(clauses):
            varrs = clause[0].split(',')
            
            neg_varrs = []
            pos_varrs = []
            
            for var in varrs:
                if var != '':
                    if var[0] == '-':
                        if var[1:] not in var_names:
                            var_names[var[1:]] =  var[1:]
                        neg_varrs.append(var_names[var[1:]])
                        
                    else:
                        if var[0:] not in var_names:
                            var_names[var[0:]] = var[0:]
                        pos_varrs.append(var_names[var[0:]])
                        
                            
            last_part = f' +cl_{idx} <= {len(neg_varrs)} \n' if clause[1] < np.inf else f' <= {len(neg_varrs) - 1} \n'           
                    
            
            file.write(f"clause_{idx}:" + ''.join([ f" -{yi}" for yi in pos_varrs]) + ''.join([ f" +{yi}" for yi in neg_varrs]) + last_part) 
                       
            
            

                
  
        file.write("\nBinaries\n")
        for idx in range(len(clauses)):
            if clauses[idx][1] < np.inf:
                file.write(f" cl_{idx}")
            
        for var_name in var_names.keys():
            file.write(f" {var_names[var_name]}")
            
        file.write('\nEnd\n')
        file.close()



def generate_instances(start_seed, end_seed, min_n, max_n, lp_dir, solveInstance, er_prob):
    
    for seed in range(start_seed, end_seed):
        
        rng = np.random.RandomState(seed)
        instance_id = rng.uniform(0,1)*100
        
        
        n = rng.randint(min_n, max_n+1)
        clauses = gen_maxcut_graph_clauses(rng,n,er_prob)
        m = len(clauses)//2
        

        instance_name = f'n={n}_m={m}_id_{instance_id:0.2f}'
        instance_path = lp_dir +  "/" + instance_name
        write_lp(clauses, instance_path + ".lp")
        print(instance_name)
        
        model = sp.Model()
        model.hideOutput()
        model.readProblem(instance_path + ".lp")
        
        if solveInstance:
            model.optimize()
            model.writeBestSol(instance_path + ".sol")  
            print(model.getNNodes())
            print(model.getSolvingTime())
            
            if model.getNNodes() <= 1:
                os.remove(instance_path+ ".lp" )
                os.remove(instance_path+ ".sol")
                
# rng = np.random.RandomState(0)       
# cl = get_clauses(rng, 2,2,20,0)
# write_lp(cl, '34.lp')
# m = sp.Model()

# m.readProblem('34.lp')
# m.optimize()
# m.writeBestSol()
