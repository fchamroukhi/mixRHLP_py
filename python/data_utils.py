#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:32:28 2018

@author: bartcus
"""

import numpy as np
import constants as const
import utils as utl

def generate_data(trace=False):
    if trace:
        utl.detect_path(const.TraceDir)
        utl.fileGlobalTrace=open(const.TraceDir + "GenerateDataTrace{0}.txt".format(const.dataName), "w")
    
    utl.globalTrace("Start data generation\n")
        
    n1 = 10 
    n2 = 10   
    n3 = 10
        
    mean_1_flou = np.loadtxt(const.datafile)
        
    y1 = np.ones((n1, 1)) * mean_1_flou.transpose() + np.random.normal(5,1,(len(mean_1_flou),n1)).transpose()+1;
    y3 = np.concatenate((np.random.normal(7,1,(80,n2)), np.random.normal(5,1,(130,n2)), np.random.normal(4,1,(140,n2)))).transpose()
    y2=np.concatenate((np.random.normal(5,1,(120,n3)),np.random.normal(7,1,(70,n3)), np.random.normal(5,1,(160,n3)))).transpose()
    data = np.concatenate((y1,y2,y3))
        
    utl.globalTrace("data size: {0}\n".format(data.shape))
    
    #write the data
    np.savetxt(const.outputGeneratedDataFileName, data)
        
    utl.globalTrace("End data generation\n")
    if trace:
        utl.fileGlobalTrace.close()
        utl.fileGlobalTrace = None    

#generate_data()