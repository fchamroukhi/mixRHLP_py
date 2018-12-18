#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 08:31:28 2018

@author: bartcus
"""
import numpy as np


class Phi():
    def __init__(self, x, p, q, n = 1):
        self.designmatrix_FRHLP(x, p, q)
        
        if n>1:
            self.designmatrix_FRHLP(x, p, q)
            self.setPhiN(n)
    
    def designmatrix_FRHLP(self, x, p, q=None):
        """
        requires:
            x - data
            p - dimension de beta (ordre de reg polynomiale)
            q (Optional) - dimension de w (ordre de reg logistique)
        ensures:
            creates the parameters phiBeta and phiW
        """
        if x.shape[0] == 1:
            x=x.T; # en vecteur
        
        order_max = p    
        if q!=None:
            order_max = max(p,q)
            
        phi=np.NaN * np.empty([len(x), order_max+1])
        for ordr in range(order_max+1):
            phi[:,ordr] = (x**ordr).transpose() # phi2w = [1 t t.^2 t.^3 t.^p;......;...]
        
        self.phiBeta = phi[:,0:p+1]; # Matrice de regresseurs pour Beta
    
        self.phiW =[];
        if q!=None:
            self.phiW = phi[:,0:q+1]; # matrice de regresseurs pour w
            
            
    def setPhiN(self, n):
        self.phiBeta = np.matlib.repmat(self.phiBeta, n, 1);
        self.phiW = np.matlib.repmat(self.phiW, n, 1);