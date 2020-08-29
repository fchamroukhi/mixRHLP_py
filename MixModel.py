#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 08:17:56 2018

@author: Faïcel Chamroukhi & Bartcus Marius
"""
import datasets as dta

class MixModel(dta.MyData):
    def __init__(self, dataFileName, G, K, p, q):
        dta.MyData.__init__(self, dataFileName) # dataset
        self.G = G # number of clusters
        self.K = K # number of regimes (i.e segments) (polynomial regression components)
        self.p = p # degree of the polynomial regressors
        self.q = q # order of the logistic regression (by default 1 for contiguous segmentation)