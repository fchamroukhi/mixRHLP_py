#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 08:22:16 2018

@author: bartcus
"""
import numpy as np
import utils as utl

class MyData():
    def __init__(self, dataFileName=None, isGenerateData = False, outputGeneratedDataFileName=None):
        self.isGenerateData = isGenerateData
        if isGenerateData:
            self.generateExampleData(dataFileName, outputGeneratedDataFileName)
        else:
            if dataFileName != None:
                self.setData(dataFileName)
    
    def setData(self, dataFileName):
        self.X = np.loadtxt(dataFileName)
        self.setDataProperties()
    
    def generateExampleData(self, dataFileName, outputGeneratedDataFileName, trace=False):
        n1 = 10 
        n2 = 10   
        n3 = 10
            
        mean_1_flou = np.loadtxt(dataFileName)
            
        y1 = np.ones((n1, 1)) * mean_1_flou.transpose() + np.random.normal(5,1,(len(mean_1_flou),n1)).transpose()+1;
        y3 = np.concatenate((np.random.normal(7,1,(80,n2)), np.random.normal(5,1,(130,n2)), np.random.normal(4,1,(140,n2)))).transpose()
        y2 = np.concatenate((np.random.normal(5,1,(120,n3)),np.random.normal(7,1,(70,n3)), np.random.normal(5,1,(160,n3)))).transpose()
        self.X = np.concatenate((y1,y2,y3))
        self.setDataProperties()
        
        #write the data
        np.savetxt(outputGeneratedDataFileName, self.X)
            
        utl.globalTrace("End data generation\n")
        if trace:
            utl.fileGlobalTrace.close()
            utl.fileGlobalTrace = None   
    
    def setDataProperties(self):
        self.n, self.m = self.X.shape
        self.XR =  np.reshape(self.X, (self.n*self.m, 1))
        #construction des matrices de regression
        self.t = np.linspace(0,1,self.m) # ou rentrer le vecteur de covariables des courbes
        self.t = np.reshape(self.t,(len(self.t),1))
        