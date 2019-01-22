#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 08:31:28 2018

Design matrices for the polynomial regression and the logistic regression

@author: Faïcel Chamroukhi
"""
import numpy as np
import numpy.matlib

class RegressionDesigner():
    def __init__(self, x, p, q, n = 1):
        self.designmatrix_FRHLP(x, p, q)
        
        if n>1:
            self.designmatrix_FRHLP(x, p, q)
            self.setPhiN(n)
    
    def designmatrix_FRHLP(self, x, p, q=None):
        """
        requires:
            x - data
            p - beta dimension (the polynomial regressions)
            q (Optional) - w dimension (the logistic regression)
        ensures:
            creates the parameters XBeta and Xw
        """
        if x.shape[0] == 1:
            x=x.T; # en vecteur
        
        order_max = p    
        if q!=None:
            order_max = max(p,q)
            
        X=np.NaN * np.empty([len(x), order_max+1])
        for ordr in range(order_max+1):
            X[:,ordr] = (x**ordr).transpose() # phi2w = [1 t t.^2 t.^3 t.^p;......;...]
        
        self.XBeta = X[:,0:p+1]; # design matrix for Beta (the polynomial regressions)
    
        self.Xw =[];
        if q!=None:
            self.Xw = X[:,0:q+1]; # design matrix for w (the logistic regression)
            
            
    def setPhiN(self, n):
        self.XBeta = np.matlib.repmat(self.XBeta, n, 1);
        self.Xw = np.matlib.repmat(self.Xw, n, 1);