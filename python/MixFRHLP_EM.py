#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:00:40 2018

@author: bartcus
"""
import numpy as np
import constants as const

class MixParam():
    def __init__(self):
        """
        ({Wg},{alpha_g}, {beta_gk},{sigma_gk}) for g=1,...,G and k=1...K. 
              1. Wg = (Wg1,...,w_gK-1) parameters of the logistic process:
                  matrix of dimension [(q+1)x(K-1)] with q the order of logistic regression.
              2. beta_g = (beta_g1,...,beta_gK) polynomial regression coefficient vectors: matrix of
                  dimension [(p+1)xK] p being the polynomial  degree.
              3. sigma_g = (sigma_g1,...,sigma_gK) : the variances for the K regmies. vector of dimension [Kx1]
              4. pi_jgk :logistic proportions for cluster g
        """
        self.Wg, self.beta_g, self.sigma_g, self.pi_jgk = pi_jgk = initialize_MixFRHLP_EM(phiBeta, phiW, try_EM);
        
        
class MixFRHLP():
    """
          1. param : the model parameters MixParam class

          2. Psi: parameter vector of the model: Psi=({Wg},{alpha_g},{beta_gk},{sigma_gk}) 
                  column vector of dim [nu x 1] with nu = nbr of free parametres
          3. h_ig = prob(curve|cluster_g) : post prob (fuzzy segmentation matrix of dim [nxG])
          4. c_ig : Hard partition obtained by the AP rule :  c_{ig} = 1
                    if and only c_i = arg max_g h_ig (g=1,...,G)
          5. klas : column vector of cluster labels
          6. tau_ijgk prob(y_{ij}|kth_segment,cluster_g), fuzzy
          segmentation for the cluster g. matrix of dimension
          [nmxK] for each g  (g=1,...,G).
          7. Ex_g: curve expectation: sum of the polynomial components beta_gk ri weighted by 
             the logitic probabilities pij_gk: Ex_g(j) = sum_{k=1}^K pi_jgk beta_gk rj, j=1,...,m. Ex_g 
              is a column vector of dimension m for each g.
          8. comp-loglik : at convergence of the EM algo
          9. stored_com-loglik : vector of stored valued of the
          comp-log-lik at each EM teration 
          
          10. BIC value = loglik - nu*log(nm)/2.
          11. ICL value = comp-loglik_star - nu*log(nm)/2.
          12. AIC value = loglik - nu.
          13. log_alphag_fg_xij 
          14. polynomials 
          15. weighted_polynomials
    """

    def __init__(self):
        self.param = None
        self.Psi = None
        self.h_ig = np.empty([const.n, const.G])
        self.c_ig = None
        self.klas = np.empty([const.n, 1])
        #todo: verify the size of tau_ijgk
        self.tau_ijgk = None
        self.Ex_g = np.empty([const.m, const.G])
        self.comp_loglik = None
        self.stored_com_loglik = np.empty([1, const.total_EM_tries])
        
        
#solution =  MixFRHLP_EM(data); 
