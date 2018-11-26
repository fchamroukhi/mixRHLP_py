#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:31:48 2018

This code contains the settings to set in order to run the Mix-EM algorithm
Each user is responsible to set these constants according to it's needs

@author: bartcus
"""
import os
import numpy as np
import datasets
"""
    Options of MixFRHLP_EM:
          1. q:  order of the logistic regression (by default 1 for convex segmentation)
          2. variance_type of the poynomial models for each cluster (free or
          common, by defalut free)
          3. init_kmeans: initialize the curve partition by Kmeans
          4. total_EM_tries :  (the solution providing the highest log-lik is chosen
          5. max_iter_EM
          6. threshold: by defalut 1e-6
          7. verbose : set to 1 for printing the "complete-log-lik"  values during
          the EM iterations (by default verbose_EM = 0)
          8. verbose_IRLS : set to 1 for printing the values of the criterion 
             optimized by IRLS at each IRLS iteration. (IRLS is used at
             each M step of the EM algorithm). (By defalut: verbose_IRLS = 0)
    """
varianceTypes=['free', 'common']


variance_type = varianceTypes[0]
total_EM_tries=2
init_kmeans=True
max_iter_EM=1000
threshold=1e-5



def setModelDimension(qq,pp,GG,KK):
    global q
    global p
    global G
    global K
    q=qq # dimension de w (ordre de reg logistique)
    p = pp # dimension de beta (ordre de reg polynomiale)
    #n_tries = 2
    G = GG # nombre de clusters
    K = KK # nombre de regimes
    
#setModelDimension(1,1,3,3)


"""
###################################################################################
    the working directories
###################################################################################    
"""
DataSetDir='data/generated/'
TraceDir = 'trace/'


if DataSetDir == None:
    DataSetDir=os.getcwd()+'/' #current directory
    
if TraceDir == None:
    TraceDir=os.getcwd()+'/' #current directory
    
"""
###################################################################################
    data sets
###################################################################################
"""
dataExetension = ".txt"

isGenerateData = False

if isGenerateData:
    dataName = 'mean_1_flou'
    generatedDataName = 'generated_data_1'
    outputGeneratedDataFileName = DataSetDir + generatedDataName + dataExetension
else:
    #Give the name of the data that needs to be loaded
    dataName=datasets.dataFileNames[0]

datafile = DataSetDir + dataName + dataExetension

data = np.loadtxt(datafile)
n, m= data.shape


