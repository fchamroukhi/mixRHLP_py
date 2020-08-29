#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:48:08 2018

@author: Faïcel Chamroukhi & Bartcus Marius
"""
import numpy as np
import RegressionDesigner as regdes
import time
import MixParam as pram
import MixStats as stats
from copy import deepcopy


## The EM Algorithm


def EM(mixModel, modelOptions):
    phi = regdes.RegressionDesigner(mixModel.t, mixModel.p, mixModel.q, mixModel.n)
    top = 0
    try_EM = 0
    best_loglik = -np.Inf
    cpu_time_all = []
    
    while(try_EM < modelOptions.n_tries):
        try_EM = try_EM+1
        print("EM try nr ",try_EM)
        start_time = time.time()
        
        # Initializations
        mixParam = pram.MixParam(mixModel, modelOptions)
        mixParam.initParam(mixModel, phi, modelOptions, try_EM)
        
        iteration = 0
        converge = False
        prev_loglik = -np.Inf
        
        mixStats = stats.MixStats(mixModel, modelOptions)
        
        while not(converge) and (iteration <= modelOptions.max_iter):
            mixStats.EStep(mixModel,mixParam,phi,modelOptions.variance_type)
            
            mixParam.MStep(mixModel, mixStats, phi, modelOptions)
            
            # FIN EM
            iteration += 1
            if (modelOptions.verbose):
                print('EM   : Iteration : {0}   log-likelihood : {1}'.format(iteration, mixStats.loglik))
            if(prev_loglik - mixStats.loglik > 1e-5):
                print('!!!!! EM log-likelihood is decreasing from {0} to {1}!'.format(prev_loglik, mixStats.loglik))
                top += 1
                if (top>20):
                    break
            
            # TEST OF CONVERGENCE
            if iteration>1:
                converge = abs((mixStats.loglik-prev_loglik)/prev_loglik)<=modelOptions.threshold
            #todo: if is nan convergence
            prev_loglik = mixStats.loglik
            mixStats.stored_loglik[iteration-1, try_EM-1] = mixStats.loglik
            
        # End of EM LOOP
        cpu_time = time.time()-start_time
        cpu_time_all.append(cpu_time)
        
        # at this point we have computed param and mixStats that contain all the information
        if (mixStats.loglik > best_loglik):
            mixStatsSolution = deepcopy(mixStats)
            mixParamSolution = deepcopy(mixParam)
            
            mixParamSolution.pi_jgk = mixParam.pi_jgk[:,0:mixModel.m,:]
            
            best_loglik = mixStats.loglik
            
        if modelOptions.n_tries > 1:
            print("max value: {0}".format(mixStatsSolution.loglik))
    mixStatsSolution.MAP()
    if modelOptions.n_tries > 1:
        print("max value: {0}".format(mixStatsSolution.loglik))
        
    # End of computation of mixStatsSolution
    mixStatsSolution.computeStats(mixModel, mixParamSolution, phi, cpu_time_all)
    
    return mixParamSolution, mixStatsSolution




## The Classification EM Algorithm



def CEM(mixModel, modelOptions):
    phi = regdes.RegressionDesigner(mixModel.t, mixModel.p, mixModel.q, mixModel.n)
    top = 0
    try_CEM = 0
    best_com_loglik = -np.Inf
    cpu_time_all = []
    
    while(try_CEM < modelOptions.n_tries):
        try_CEM = try_CEM+1
        print("CEM try nr ",try_CEM)
        start_time = time.time()
        
        # Initializations
        mixParam = pram.MixParam(mixModel, modelOptions)
        mixParam.initParam(mixModel, phi, modelOptions, try_CEM)
        
        iteration = 0
        converge = False
        prev_comp_loglik = -np.Inf
        reg_irls = 0
        
        mixStats = stats.MixStats(mixModel, modelOptions)
        
        while not(converge) and (iteration <= modelOptions.max_iter):
            mixStats.EStep(mixModel,mixParam,phi,modelOptions.variance_type)
            mixStats.CStep(reg_irls)
            reg_irls, good_segmentation = mixParam.CMStep(mixModel, mixStats, phi, modelOptions)
            
            if (good_segmentation==False):
                try_CEM = try_CEM - 1
                break
            # FIN EM
            iteration += 1
            if (modelOptions.verbose):
                print('CEM   : Iteration : {0}   complete log-likelihood : {1}'.format(iteration, mixStats.comp_loglik))
            if(prev_comp_loglik - mixStats.comp_loglik > 1e-5):
                print('!!!!! CEM complete log-likelihood is decreasing from {0} to {1}!'.format(prev_comp_loglik, mixStats.comp_loglik))
                top += 1
                if (top>20):
                    break
            
            # TEST OF CONVERGENCE
            if iteration>1:
                converge = abs((mixStats.comp_loglik-prev_comp_loglik)/prev_comp_loglik)<=modelOptions.threshold
            #todo: if is nan convergence
            prev_comp_loglik = mixStats.comp_loglik
            mixStats.stored_com_loglik[iteration-1, try_CEM-1] = mixStats.comp_loglik
            
        # End of CEM LOOP
        cpu_time = time.time()-start_time
        cpu_time_all.append(cpu_time)
        
        # at this point we have computed param and mixStats that contains all the information
        if (mixStats.comp_loglik > best_com_loglik):
            mixStatsSolution = deepcopy(mixStats)
            mixParamSolution = deepcopy(mixParam)
            
            mixParamSolution.pi_jgk = mixParam.pi_jgk[:,0:mixModel.m,:]
            
            best_com_loglik = mixStats.comp_loglik
        if modelOptions.n_tries > 1:
            print("max value: {0}".format(mixStatsSolution.comp_loglik))
    
    
    if modelOptions.n_tries > 1:
        print("max value: {0}".format(mixStatsSolution.comp_loglik))
        
    # End of computation of mixStatsSolution
    mixStatsSolution.computeStats(mixModel, mixParamSolution, phi, cpu_time_all)
    
    return mixParamSolution, mixStatsSolution
        