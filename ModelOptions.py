#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 08:24:48 2018

@author: Faïcel Chamroukhi & Bartcus Marius
"""

class ModelOptions():
    def __init__(self, n_tries, max_iter, threshold, verbose, verbose_IRLS, init_kmeans, variance_type):
        self.n_tries = n_tries # number of algorithm running (different initializations)
        self.max_iter = max_iter
        self.threshold = threshold #to ckeck the improvement of the loglikelihood 
        self.verbose = verbose
        self.verbose_IRLS = verbose_IRLS
        self.init_kmeans = init_kmeans
        self.variance_type = variance_type #(for a homoskedastic or a heteroskedastic model)