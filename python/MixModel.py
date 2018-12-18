#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 08:17:56 2018

@author: bartcus
"""
import datasets as dta

class MixModel(dta.MyData):
    def __init__(self, dataFileName, G, K, p, q):
        dta.MyData.__init__(self, dataFileName)
        self.G = G
        self.K = K
        self.p = p
        self.q = q