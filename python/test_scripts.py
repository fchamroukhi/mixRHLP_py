#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 08:41:10 2018

@author: bartcus
"""

#import datasets as dta
import MixModel as model
import Phi
import ModelOptions as options
import enums
import MixParam as pram
from copy import deepcopy

def testModelCreation():
    dataFileName = "data/generated/generated_data_1.txt"
    #mixData = dta.MyData(dataFileName)
    G = 3; # nombre de clusters
    K = 3; #nombre de regimes
    p = 1; #dimension de beta (ordre de reg polynomiale)
    q = 1; #dimension de w (ordre de reg logistique)
    mixModel = model.MixModel(dataFileName, G, K, p, q)
    
    phi = Phi.Phi(mixModel.t, mixModel.p, mixModel.q, mixModel.n)
    
    
    n_tries=1
    max_iter=1000
    threshold = 1e-5
    verbose = True
    verbose_IRLS = True
    init_kmeans = True
    mixOptions = options.ModelOptions(n_tries, max_iter, threshold, verbose, verbose_IRLS, init_kmeans, enums.variance_types.common)
    
    param = pram.MixParam(mixModel, mixOptions)
    param.initParam(mixModel, phi, mixOptions, try_algo = 1)
    return param

param = testModelCreation()

param2 = deepcopy(param)