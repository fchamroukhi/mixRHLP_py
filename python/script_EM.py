#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:20:37 2018

@author: bartcus
"""
import MixModel as model
import ModelOptions as options
import enums
import ModelLearner as learner

dataFileName = "data/generated/generated_data_1.txt"
G = 3; # nombre de clusters
K = 3; #nombre de regimes
p = 1; #dimension de beta (ordre de reg polynomiale)
q = 1; #dimension de w (ordre de reg logistique)
mixModel = model.MixModel(dataFileName, G, K, p, q)

n_tries=1
max_iter=1000
threshold = 1e-5
verbose = True
verbose_IRLS = True
init_kmeans = True
modelOptions = options.ModelOptions(n_tries, max_iter, threshold, verbose, verbose_IRLS, init_kmeans, enums.variance_types.free)

mixParamSolution, mixStatsSolution = learner.EM(mixModel, modelOptions)
mixStatsSolution.showDataClusterSegmentation(mixModel, mixParamSolution)


#mixParamSolution, mixStatsSolution = learner.CEM(mixModel, modelOptions)
#mixStatsSolution.showDataClusterSegmentation(mixModel, mixParamSolution)