#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:05:18 2018

@author: bartcus
"""

import numpy as np

def main():
    """ simulated data """
    n1 = 10 
    n2 = 10   
    n3 = 10
    datafile = "data/mean_1_flou.txt"
    mean_1_flou = np.loadtxt(datafile)
    
    y1 = np.ones((n1, 1)) * mean_1_flou.transpose() + np.random.normal(5,1,(len(mean_1_flou),n1)).transpose()+1;
    
    y3 = np.concatenate((np.random.normal(7,1,(80,n2)), np.random.normal(5,1,(130,n2)), np.random.normal(4,1,(140,n2)))).transpose()
    
    y2=np.concatenate((np.random.normal(5,1,(120,n3)),np.random.normal(7,1,(70,n3)), np.random.normal(5,1,(160,n3)))).transpose()
    
    data = np.concatenate((y1,y2,y3))
    #data.shape
    
    
    G = 3;# nombre de clusters
    K = 3;# nombre de regimes
    p = 1;# dimension de beta (ordre de reg polynomiale)
    q = 1;# dimension de w (ordre de reg logistique)
    
    #type_variance = 'common';
    type_variance = 'free';
    n_tries = 2;
    max_iter = 1000;
    init_kmeans = 1;
    threshold = 1e-5;
    verbose = 1; 
    verbose_IRLS = 0;

    n,m = data.shape
    
    solution =  MixFRHLP_EM(data, G , K, p, q, type_variance, init_kmeans, n_tries, max_iter, threshold, verbose, verbose_IRLS);        
