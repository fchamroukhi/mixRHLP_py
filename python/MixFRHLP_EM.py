#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:00:40 2018

@author: bartcus
"""
import time
import sys
import numpy as np
import utils as utl
import constants as const
import MixFRHLP_Parameters as mixParam
        
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
        self.param = np.NaN
        self.Psi = np.NaN
        self.h_ig = np.NaN*np.empty([const.n, const.G])
        self.c_ig = np.NaN
        self.klas = np.NaN*np.empty([const.n, 1])
        self.tau_ijgk = np.NaN*np.empty([const.G, const.n * const.m, const.K])
        self.Ex_g = np.NaN*np.empty([const.m, const.G])
        self.comp_loglik = np.NaN
        self.stored_com_loglik = np.NaN*np.empty([1, const.total_EM_tries])
        
    def fit_EM(self, trace=True):
        """
            main algorithm
        """
        if trace:
            utl.detect_path(const.TraceDir)
            utl.fileGlobalTrace=open(const.TraceDir + "FitEM_Trace{0}.txt".format(const.dataName), "w")
        utl.globalTrace("Start EM\n")
        
        # 1. Construction des matrices de regression
        x = np.linspace(0,1,const.m) # ou rentrer le vecteur de covariables des courbes
        # 2. pour 1 courbe
        phiBeta, phiW = utl.designmatrix_FRHLP(x, const.p, const.q);
        #pour les n courbes (regularly sampled)
        phiBeta = np.matlib.repmat(phiBeta, const.n, 1);
        phiW = np.matlib.repmat(phiW, const.n, 1);
        
        X = np.reshape(const.data.T,(const.n*const.m, 1))
        
        top=0
        try_EM = 0
        best_loglik = -np.Inf
        cputime_total = []
        
        while try_EM < const.total_EM_tries:
            try_EM+=1
            utl.globalTrace("EM try: {0}\n".format(try_EM))
            start_time = time.time()
            
            #initialization param
            self.param = mixParam.MixParam()
            self.param.initialize_MixFRHLP_EM(phiBeta, phiW, try_EM)
            
            iteration = 0; 
            converge = False
            prev_loglik = -np.Inf
            
            #EM
            self.tau_ijgk = np.zeros((const.G, const.n*const.m, const.K)) # segments post prob  
            log_tau_ijgk = np.zeros((const.G, const.n*const.m, const.K))
            
            log_fg_xij = np.zeros((const.n,const.G))
            log_alphag_fg_xij = np.zeros((const.n,const.G))
            
            
            while not(converge) and (iteration<= const.max_iter_EM):
                """
                E-Step
                """
                self.__EStep(X, phiBeta, log_tau_ijgk, log_fg_xij, log_alphag_fg_xij)
                """
                M-Step
                """
                self.__MStep()
            
            cpu_time = time.time()-start_time
            cputime_total.append(cpu_time)
            
            
        utl.globalTrace("End EM\n")
        if trace:
            utl.fileGlobalTrace.close()
            utl.fileGlobalTrace = None
            
        
    def __EStep(self, X, phiBeta, log_tau_ijgk, log_fg_xij, log_alphag_fg_xij):
        for g in range(0,const.G):
            alpha_g = self.param.alpha_g
            beta_g = self.param.beta_g[g,:,:]
            Wg = self.param.Wg[g,:,:]
            pi_jgk = self.param.pi_jgk[g,:,:]
            
            log_pijgk_fgk_xij = np.zeros((const.n*const.m,const.K))
            for k in range(0,const.K):
                beta_gk = beta_g[:,k]
                if const.variance_type.lower() == 'common':
                    sgk = self.param.sigma_g(g)
                else:
                    sgk = self.param.sigma_g[k,g]
                
                temp = phiBeta@beta_gk
                temp = temp.reshape((len(temp), 1))
                z=((X-temp)**2)/sgk;
                
                temp = np.array([np.log(pi_jgk[:,k]) - 0.5*(np.log(2*np.pi) + np.log(sgk))]).T - 0.5*z
                log_pijgk_fgk_xij[:,k] = temp[0]; #pdf cond à c_i = g et z_i = k de xij
                
                
            log_pijgk_fgk_xij = np.minimum(log_pijgk_fgk_xij,np.log(sys.float_info.max))
            log_pijgk_fgk_xij = np.maximum(log_pijgk_fgk_xij,np.log(sys.float_info.min))
            
            pijgk_fgk_xij = np.exp(log_pijgk_fgk_xij);
            sumk_pijgk_fgk_xij = np.array([pijgk_fgk_xij.sum(axis = 1)]) # sum over k
            sumk_pijgk_fgk_xij = sumk_pijgk_fgk_xij.T 
            log_sumk_pijgk_fgk_xij  = np.log(sumk_pijgk_fgk_xij) #[nxm x 1]
            
            log_tau_ijgk[g,:,:] = log_pijgk_fgk_xij - log_sumk_pijgk_fgk_xij * np.ones((1,const.K));
            self.tau_ijgk[g,:,:] = np.exp(utl.log_normalize(log_tau_ijgk[g,:,:])); 
            
            temp = log_sumk_pijgk_fgk_xij.reshape(const.m,const.n).T
            log_fg_xij[:,g] = temp.sum(axis = 1); #[n x 1]:  sum over j=1,...,m: fg_xij = prod_j sum_k pi_{jgk} N(x_{ij},mu_{gk},s_{gk))
            log_alphag_fg_xij[:,g] = np.log(alpha_g[g]) + log_fg_xij[:,g]; # [nxg] 
            
        log_alphag_fg_xij = np.minimum(log_alphag_fg_xij,np.log(sys.float_info.max));
        log_alphag_fg_xij = np.maximum(log_alphag_fg_xij,np.log(sys.float_info.min));

        # cluster posterior probabilities p(c_i=g|X)
        h_ig = np.exp(utl.log_normalize(log_alphag_fg_xij)); 
        
        # log-likelihood
        temp = np.exp(log_alphag_fg_xij)
        loglik = sum(np.log(temp.sum(axis = 1)));
        
        return loglik, h_ig
#solution =  MixFRHLP_EM(data); 
