#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:49:03 2018

@author: bartcus
"""
import MixFRHLP_EM as mix
import constants as const
import utils as utl

def main_script():
    p=1
    q=1
    G=3
    K=3
    const.setModelDimension(p,q,G,K)
    solution =  mix.MixFRHLP(); 
    solution.fit_EM()
    return solution

s = main_script()

utl.showResults(const.data, s)