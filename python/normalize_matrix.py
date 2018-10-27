#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:29:21 2018

@author: bartcus
"""
import numpy as np
from sklearn.preprocessing import normalize

def normalize_matrix(matrix):
    #Scikit-learn normalize function that lets you apply various normalizations. 
    #The "make it sum to 1" is the L1 norm
    normed_matrix = normalize(matrix, axis=1, norm='l1')  
    return normed_matrix


def test_main():
    matrix = np.arange(0,27,3).reshape(3,3).astype(np.float64)
    #array([[  0.,   3.,   6.],
    #   [  9.,  12.,  15.],
    #   [ 18.,  21.,  24.]])
    print(normalize_matrix(matrix))
    #[[ 0.          0.33333333  0.66666667]
    #[ 0.25        0.33333333  0.41666667]
    #[ 0.28571429  0.33333333  0.38095238]]
    
    
    
    
#test_main()