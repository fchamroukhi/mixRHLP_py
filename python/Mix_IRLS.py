#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:04:19 2018

@author: bartcus
"""

class IRLS():
    def __init__(self, cluster_weights, tauijk, phiW, Wg_init):
        self.cluster_weights = cluster_weights
        self.tauijk = tauijk
        self.phiW = phiW
        self.Wg_init = Wg_init