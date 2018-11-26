#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:49:03 2018

@author: bartcus
"""
import matplotlib.pyplot as plt
import MixFRHLP_EM as mix

def main_script():
    solution =  mix.MixFRHLP(); 
    solution.fit_EM()
    return solution

s = main_script()
plt.plot(s.bestSolution.stored_loglik)