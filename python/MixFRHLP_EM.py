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
import default_constants as defConst
import MixFRHLP_Parameters as mixParam
import Mix_IRLS as mixirls
import matplotlib.pyplot as plt


class MixFRHLPSolution():
    def __init__(self, param, Psi, h_ig, tau_ijgk, Ex_g, loglik, stored_loglik, log_alphag_fg_xij):
        self.param = param
        self.param.pi_jgk = self.param.pi_jgk[:,0:const.m,:]
        self.Psi = Psi
        self.h_ig = h_ig
        self.tau_ijgk = tau_ijgk
        self.Ex_g = Ex_g
        self.loglik = loglik
        self.stored_loglik = stored_loglik
        self.log_alphag_fg_xij = log_alphag_fg_xij
        
        #klas and c_ig are recomputed by MAP
        self.klas = np.NaN*np.empty([const.n, 1])
        self.c_ig = np.NaN
        
        self.polynomials = np.NaN*np.empty([const.G, const.m, const.K])
        self.weighted_polynomials = np.NaN*np.empty([const.G, const.m, const.K])
        
    def setCompleteSolution(self, phiBeta, cputime_total):
        for g in range(0,const.G):
            self.polynomials[g,:,:] = phiBeta[0:const.m,:]@self.param.beta_g[g,:,:]
            print(self.param.pi_jgk[g,:,:].shape)
            self.weighted_polynomials[g,:,:] = self.param.pi_jgk[g,:,:]*self.polynomials[g,:,:]
            self.Ex_g[:,g] = self.weighted_polynomials[g,:,:].sum(axis=1); 
        
        self.Ex_g = self.Ex_g[0:const.m,:] 
        self.cputime = np.mean(cputime_total)
        
        nu = len(self.Psi);
        # BIC AIC et ICL*
        self.BIC = self.loglik - (nu*np.log(const.n)/2) #n*m/2!
        self.AIC = self.loglik - nu
        # ICL*             
        # Compute the comp-log-lik 
        cig_log_alphag_fg_xij = (self.c_ig)*(self.log_alphag_fg_xij);
        comp_loglik = sum(cig_log_alphag_fg_xij.sum(axis=1)) 

        self.ICL = comp_loglik - nu*np.log(const.n)/2 #n*m/2!

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
        self.loglik = np.NaN
        self.stored_loglik = np.NaN*np.empty([const.total_EM_tries,1])
        self.comp_loglik = np.NaN
        self.stored_com_loglik = np.NaN*np.empty([1, const.total_EM_tries])
        self.log_alphag_fg_xij = np.zeros((const.n,const.G))
        
        #self.mixSolution = MixFRHLPSolution()
        
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
            self.log_alphag_fg_xij = np.zeros((const.n,const.G))
            
            
            while not(converge) and (iteration<= const.max_iter_EM):
                """
                E-Step
                """
                self.__EStep(X, phiBeta, log_tau_ijgk, log_fg_xij)
                """
                M-Step
                """
                self.__MStep(X, phiBeta, phiW)
                
                #FIN EM
                
                iteration+=1
                
                utl.globalTrace('EM   : Iteration : {0}   log-likelihood : {1} \n'.format(iteration, self.loglik))
                if prev_loglik-self.loglik>1e-5:
                    utl.globalTrace('!!!!! EM log-likelihood is decreasing from {0} to {1}!\n'.format(prev_loglik, self.loglik))
                    top+=1
                    if top>20:
                        break
                converge = abs((self.loglik-prev_loglik)/prev_loglik)<=const.threshold
                
                prev_loglik = self.loglik;
                self.stored_loglik[iteration-1]=self.loglik
            
            cpu_time = time.time()-start_time
            cputime_total.append(cpu_time)
            
            self.Psi = np.array([self.param.alpha_g.T.ravel(), self.param.Wg.T.ravel(), self.param.beta_g.T.ravel(), self.param.sigma_g.T.ravel()])
            if self.loglik > best_loglik:
                self.bestSolution = MixFRHLPSolution(self.param, self.Psi, self.h_ig, self.tau_ijgk, self.Ex_g, self.loglik, self.stored_loglik, self.log_alphag_fg_xij)
                best_loglik = self.loglik
                
            if const.total_EM_tries>1:
                utl.globalTrace('max value: {0} \n'.format(self.loglik))
        
              
        self.bestSolution.klas, self.bestSolution.c_ig = utl.MAP(self.bestSolution.h_ig); # c_ig the hard partition of the curves
        
        
        if const.total_EM_tries>1:
            utl.globalTrace('max value: {0} \n'.format(self.loglik))
        
        self.bestSolution.setCompleteSolution(phiBeta, cpu_time)
        

        utl.globalTrace("End EM\n")
        if trace:
            utl.fileGlobalTrace.close()
            utl.fileGlobalTrace = None
            
        
    def __EStep(self, X, phiBeta, log_tau_ijgk, log_fg_xij):
        """
        E-step
        """
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
                    #todo: verify
                    sgk = self.param.sigma_g[g,k]
                
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
            
            log_tau_ijgk[g,:,:] = log_pijgk_fgk_xij - log_sumk_pijgk_fgk_xij * np.ones((1,const.K))
            self.tau_ijgk[g,:,:] = np.exp(utl.log_normalize(log_tau_ijgk[g,:,:]))
            
            temp = log_sumk_pijgk_fgk_xij.reshape(const.m,const.n).T
            log_fg_xij[:,g] = temp.sum(axis = 1) #[n x 1]:  sum over j=1,...,m: fg_xij = prod_j sum_k pi_{jgk} N(x_{ij},mu_{gk},s_{gk))
            self.log_alphag_fg_xij[:,g] = np.log(alpha_g[g]) + log_fg_xij[:,g] # [nxg] 
            
        self.log_alphag_fg_xij = np.minimum(self.log_alphag_fg_xij,np.log(sys.float_info.max))
        self.log_alphag_fg_xij = np.maximum(self.log_alphag_fg_xij,np.log(sys.float_info.min))

        # cluster posterior probabilities p(c_i=g|X)
        self.h_ig = np.exp(utl.log_normalize(self.log_alphag_fg_xij))
        
        # log-likelihood
        temp = np.exp(self.log_alphag_fg_xij)
        self.loglik = sum(np.log(temp.sum(axis = 1)))
        
    
    
    def __MStep(self, X, phiBeta, phiW):
        """
        M-step
        """
        # Maximization w.r.t alpha_g
        self.param.alpha_g = np.array([self.h_ig.sum(axis=0)]).T/const.n
        
        # Maximization w.r.t betagk et sigmagk
        for g in range(0,const.G):
            temp =  np.matlib.repmat(self.h_ig[:,g],const.m,1) # [m x n]
            cluster_weights = temp.T.reshape(temp.size,1)
            tauijk = self.tau_ijgk[g,:,:] #[(nxm) x K]
            if const.variance_type.lower() == 'common':  
                s = 0 
            else:
                sigma_gk = np.zeros((const.K,1))
            
            beta_gk = np.NaN * np.empty([const.p +1, const.K])
            for k in range(0,const.K):
                segment_weights = np.array([tauijk[:,k]]).T #poids du kieme segment   pour le cluster g  
                # poids pour avoir K segments floues du gieme cluster flou 
                phigk = (np.sqrt(cluster_weights*segment_weights)*np.ones((1,const.p+1)))*phiBeta #[(n*m)*(p+1)]
                Xgk = np.sqrt(cluster_weights*segment_weights)*X
                
                # maximization w.r.t beta_gk: Weighted least squares
                temp = np.linalg.inv(phigk.T@phigk + defConst.eps*np.eye(const.p+1))@phigk.T@Xgk
                beta_gk[:,k] = temp.ravel() # Maximization w.r.t betagk
                
                # Maximization w.r.t au sigma_gk :   
                if const.variance_type.lower() == 'common':
                    sk = sum((Xgk-phigk@beta_gk[:,k])**2)
                    s = s+sk;
                    sigma_gk = s/sum(sum((cluster_weights@np.ones((1,const.K)))*tauijk))
                else:
                    temp = phigk@np.array([beta_gk[:,k]]).T
                    sigma_gk[k]= sum((Xgk-temp)**2)/(sum(cluster_weights*segment_weights))
                    
            self.param.beta_g[g,:,:] = beta_gk
            self.param.sigma_g[g,:] = list(sigma_gk)
            
            """
            Maximization w.r.t W 
            IRLS : Regression logistique multinomiale pondérée par cluster
            """
            Wg_init = self.param.Wg[g,:,:]
            irls = mixirls.IRLS()
            irls.runIRLS(cluster_weights, tauijk, phiW, Wg_init)
            
#            a=irls.piik[0:const.m,:]
#            plt.plot(a)
#            plt.show()
            
            self.param.Wg[g,:,:]=irls.wk;             
            self.param.pi_jgk[g,:,:] = np.matlib.repmat(irls.piik[0:const.m,:],const.n,1); 
            
def main_script():
    solution =  MixFRHLP(); 
    solution.fit_EM()
    return solution

s = main_script()