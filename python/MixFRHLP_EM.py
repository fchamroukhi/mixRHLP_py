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



class MixFRHLPSolution():
    def __init__(self, param, Psi, h_ig, tau_ijgk, Ex_g, loglik, stored_loglik, comp_loglik, stored_comp_loglik, log_alphag_fg_xij):
        self.param = param
        self.param.pi_jgk = self.param.pi_jgk[:,0:const.m,:]
        self.Psi = Psi
        self.h_ig = h_ig
        self.tau_ijgk = tau_ijgk
        self.Ex_g = Ex_g
        self.loglik = loglik
        self.stored_loglik = stored_loglik
        self.comp_loglik = comp_loglik
        self.stored_comp_loglik = stored_comp_loglik
        self.log_alphag_fg_xij = log_alphag_fg_xij
        
        if const.alg.lower() == 'cem':
            self.loglik = sum(np.log(np.exp(log_alphag_fg_xij).sum(1)));
        
        #klas and c_ig are recomputed by MAP
        self.klas = np.NaN*np.empty([const.n, 1])
        self.c_ig = np.NaN
        
        self.polynomials = np.NaN*np.empty([const.G, const.m, const.K])
        self.weighted_polynomials = np.NaN*np.empty([const.G, const.m, const.K])
        
    def setCompleteSolution(self, phiBeta, cputime_total):
        for g in range(0,const.G):
            self.polynomials[g,:,:] = phiBeta[0:const.m,:]@self.param.beta_g[g,:,:]
            
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
        self.stored_loglik = np.NaN*np.empty([const.max_iter_EM, const.total_EM_tries])
        self.comp_loglik = np.NaN
        self.stored_com_loglik = np.NaN*np.empty([const.max_iter_EM, const.total_EM_tries])
        
        
        self.log_alphag_fg_xij = np.zeros((const.n,const.G))
        
        
        self.cputime_total = []
        #self.mixSolution = MixFRHLPSolution()
        
    def fit_EM(self, trace=True):
        """
            main algorithm
        """
        if trace:
            utl.detect_path(const.TraceDir)
            if const.alg.lower() == 'cem':
                utl.fileGlobalTrace=open(const.TraceDir + "FitCEM_Trace{0}.txt".format(const.dataName), "w")
            if const.alg.lower() == 'em':
                utl.fileGlobalTrace=open(const.TraceDir + "FitEM_Trace{0}.txt".format(const.dataName), "w")
        utl.globalTrace("Start EM\n")
        
        # 1. Construction des matrices de regression
        x = np.linspace(0,1,const.m) # ou rentrer le vecteur de covariables des courbes
        x = np.reshape(x,(len(x),1))
        # 2. pour 1 courbe
        phiBeta, phiW = utl.designmatrix_FRHLP(x, const.p, const.q);
        #pour les n courbes (regularly sampled)
        phiBeta = np.matlib.repmat(phiBeta, const.n, 1);
        phiW = np.matlib.repmat(phiW, const.n, 1);
        
        X = np.reshape(const.data,(const.n*const.m, 1))
        
        top=0
        try_EM = 0
        best_criterion = -np.Inf
        
        
        while try_EM < const.total_EM_tries:
            try_EM+=1
            utl.globalTrace("EM try: {0}\n".format(try_EM))
            start_time = time.time()
            
            #initialization param
            self.param = mixParam.MixParam()
            self.param.initialize_MixFRHLP_EM(phiBeta, phiW, try_EM)
            
            #todo: delete
            #self.param.sigma_g = np.array([[1.1962,1.2821,0.9436],[1.2485,1.1025,1.5289],[0.9995,0.9589,0.9512]])
            #self.param.beta_g[0,:,:] = np.array([[6.7429,1.3500 ,6.9402],[-4.1544,7.6295,-0.2227]])
            #self.param.beta_g[1,:,:] = np.array([[7.5825,6.0601 ,3.9628],[ -7.2452, -2.6757, 0.0313]])
            #self.param.betdo the followin (I think if I put the whole context of what I am doing, it will be ma_g[2,:,:] = np.array([[4.9583,    9.5519,    4.8896],[ 0.1544,   -6.8148,    0.1226]])
            
            
            iteration = 0; 
            converge = False
            prev_criterion = -np.Inf
            
            #EM
            self.tau_ijgk = np.zeros((const.G, const.n*const.m, const.K)) # segments post prob  
            log_tau_ijgk = np.zeros((const.G, const.n*const.m, const.K))
            
            log_fg_xij = np.zeros((const.n,const.G))
            self.log_alphag_fg_xij = np.zeros((const.n,const.G))
            
            
            while not(converge) and (iteration<= const.max_iter_EM):
                iteration+=1
                """
                E-Step
                """
                self.__EStep(X, phiBeta, log_tau_ijgk, log_fg_xij)
                
                
                
                
                irls = mixirls.IRLS()
                
                
                
                """
                CM-Step
                """
                if const.alg.lower() == 'cem':
                    self.__CMStep(X, phiBeta, phiW, irls)
                    utl.globalTrace('CEM   : Iteration : {0}   criterion : {1} \n'.format(iteration, self.comp_loglik))
                    
                    if prev_criterion-self.comp_loglik>1e-5:
                        utl.globalTrace('!!!!! EM log-likelihood is decreasing from {0} to {1}!\n'.format(prev_criterion, self.loglik))
                        top+=1
                        if top>20:
                            break
                    
                    converge = abs((self.comp_loglik-prev_criterion)/prev_criterion)<=const.threshold
                    prev_criterion = self.comp_loglik
                    self.stored_com_loglik[iteration-1, try_EM-1]=self.comp_loglik
                 
                """
                M-Step
                """
                if const.alg.lower() == 'em':
                    self.__MStep(X, phiBeta, phiW, irls)
                    utl.globalTrace('EM   : Iteration : {0}   log-likelihood : {1} \n'.format(iteration, self.loglik))
                
                
                
                    if prev_criterion-self.loglik>1e-5:
                        utl.globalTrace('!!!!! EM log-likelihood is decreasing from {0} to {1}!\n'.format(prev_criterion, self.loglik))
                        top+=1
                        if top>20:
                            break
                    
                    converge = abs((self.loglik-prev_criterion)/prev_criterion)<=const.threshold
                    prev_criterion = self.loglik;
                    self.stored_loglik[iteration-1, try_EM-1]=self.loglik
                
                
            
            cpu_time = time.time()-start_time
            self.cputime_total.append(cpu_time)
            
            self.Psi = np.array([self.param.alpha_g.T.ravel(), self.param.Wg.T.ravel(), self.param.beta_g.T.ravel(), self.param.sigma_g.T.ravel()])
            
            if const.alg.lower() == 'cem':
                crit=self.comp_loglik
            if const.alg.lower() == 'em':
                crit=self.loglik
                
            if crit > best_criterion:
                self.bestSolution = MixFRHLPSolution(self.param, self.Psi, self.h_ig, self.tau_ijgk, self.Ex_g, self.loglik, self.stored_loglik, self.comp_loglik, self.stored_com_loglik, self.log_alphag_fg_xij)
                best_criterion = crit
                
            if const.total_EM_tries>1:
                utl.globalTrace('max value (one try): {0} \n'.format(self.loglik))
        
        
        
        
        
        
        if const.alg.lower() == 'cem':  
            self.bestSolution.klas = self.klas
            self.bestSolution.c_ig = self.c_ig
        if const.alg.lower() == 'em':      
            self.bestSolution.klas, self.bestSolution.c_ig = utl.MAP(self.bestSolution.h_ig); # c_ig the hard partition of the curves
        
        
        if const.total_EM_tries>1:
            utl.globalTrace('max value (best solution): {0} \n'.format(self.bestSolution.loglik))
        
        self.bestSolution.setCompleteSolution(phiBeta, cpu_time)
        

        utl.globalTrace("End EM\n")
        if trace:
            utl.fileGlobalTrace.close()
            utl.fileGlobalTrace = None
            
    
    def __EStep(self, X, phiBeta, log_tau_ijgk, log_fg_xij):
        """
        E-step
        """
                
        alpha_g = self.param.alpha_g
        for g in range(0,const.G):
            
            beta_g = self.param.beta_g[g,:,:]
            #Wg = self.param.Wg[g,:,:]
            pi_jgk = self.param.pi_jgk[g,:,:]
            
            log_pijgk_fgk_xij = np.zeros((const.n*const.m,const.K))
            for k in range(0,const.K):
                beta_gk = beta_g[:,k]
                if const.variance_type.lower() == 'common':
                    sgk = self.param.sigma_g(g)
                else:
                    #todo: verify
                    sgk = self.param.sigma_g[k,g]
                
                temp = phiBeta@beta_gk
                temp = temp.reshape((len(temp), 1))
                z=((X-temp)**2)/sgk;
                
                temp = np.array([np.log(pi_jgk[:,k]) - 0.5*(np.log(2*np.pi) + np.log(sgk))]).T - 0.5*z
                log_pijgk_fgk_xij[:,k] = temp.T #pdf cond à c_i = g et z_i = k de xij
                
                
            log_pijgk_fgk_xij = np.minimum(log_pijgk_fgk_xij,np.log(sys.float_info.max))
            log_pijgk_fgk_xij = np.maximum(log_pijgk_fgk_xij,np.log(sys.float_info.min))
            
            pijgk_fgk_xij = np.exp(log_pijgk_fgk_xij);
            sumk_pijgk_fgk_xij = np.array([pijgk_fgk_xij.sum(axis = 1)]) # sum over k
            sumk_pijgk_fgk_xij = sumk_pijgk_fgk_xij.T 
            log_sumk_pijgk_fgk_xij  = np.log(sumk_pijgk_fgk_xij) #[nxm x 1]
            
            log_tau_ijgk[g,:,:] = log_pijgk_fgk_xij - log_sumk_pijgk_fgk_xij @ np.ones((1,const.K))
            self.tau_ijgk[g,:,:] = np.exp(utl.log_normalize(log_tau_ijgk[g,:,:]))
            
            temp = np.reshape(log_sumk_pijgk_fgk_xij.T,(const.n, const.m))
            log_fg_xij[:,g] = temp.sum(axis = 1) #[n x 1]:  sum over j=1,...,m: fg_xij = prod_j sum_k pi_{jgk} N(x_{ij},mu_{gk},s_{gk))
            self.log_alphag_fg_xij[:,g] = np.log(alpha_g[g]) + log_fg_xij[:,g] # [nxg] 
            
        self.log_alphag_fg_xij = np.minimum(self.log_alphag_fg_xij,np.log(sys.float_info.max))
        self.log_alphag_fg_xij = np.maximum(self.log_alphag_fg_xij,np.log(sys.float_info.min))

        
        
    
    
    def __MStep(self, X, phiBeta, phiW, irls):
        """
        M-step
        """
        # cluster posterior probabilities p(c_i=g|X)
        self.h_ig = np.exp(utl.log_normalize(self.log_alphag_fg_xij))
        #print(self.h_ig.sum(1))
        #wait = input("PRESS ENTER TO CONTINUE.")
        # log-likelihood
        temp = np.exp(self.log_alphag_fg_xij)
        self.loglik = sum(np.log(temp.sum(axis = 1)))
        #print(self.loglik)
        #wait = input("PRESS ENTER TO CONTINUE.")
        
        
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
                phigk = (np.sqrt(cluster_weights*segment_weights)@np.ones((1,const.p+1)))*phiBeta #[(n*m)*(p+1)]
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
            
            
            irls.runIRLS(tauijk, phiW, Wg_init, cluster_weights)
            
            
            self.param.Wg[g,:,:]=irls.wk;             
            self.param.pi_jgk[g,:,:] = np.matlib.repmat(irls.piik[0:const.m,:],const.n,1); 
            
    def __CMStep(self, X, phiBeta, phiW, irls):
        # cluster posterior probabilities p(c_i=g|X)
        self.h_ig = np.exp(utl.log_normalize(self.log_alphag_fg_xij))
        
        [self.klas, self.c_ig] = utl.MAP(self.h_ig); # c_ig the hard partition of the curves 
 
        #Compute the optimized criterion  
        cig_log_alphag_fg_xij = self.c_ig*self.log_alphag_fg_xij;
        self.comp_loglik = sum(cig_log_alphag_fg_xij.sum(axis=1)) +  irls.reg_irls ; 
        
        self.param.alpha_g = self.c_ig.sum(0).T/const.n;
        
        # Maximization w.r.t betagk et sigmagk
        cluster_labels =  np.matlib.repmat(self.klas,1,const.m).T # [m x n]
        cluster_labels = cluster_labels.T.ravel()
        # Maximization w.r.t betagk et sigmagk
        for g in range(0,const.G):
            Xg = X[cluster_labels==g ,:]; # cluster g (found from a hard clustering)
            tauijk = self.tau_ijgk[g,cluster_labels==g,:]
            if const.variance_type.lower() == 'common':  
                s = 0 
            else:
                sigma_gk = np.zeros((const.K,1))
                
            beta_gk = np.NaN * np.empty([const.p +1, const.K])
            for k in range(0,const.K):
                segment_weights = np.array([tauijk[:,k]]).T #poids du kieme segment   pour le cluster g  
                phigk = (np.sqrt(segment_weights)@np.ones((1,const.p+1)))*phiBeta[cluster_labels==g,:] #[(n*m)*(p+1)]
                Xgk = np.sqrt(segment_weights)*Xg
                # maximization w.r.t beta_gk: Weighted least squares 
                temp = np.linalg.inv(phigk.T@phigk + defConst.eps*np.eye(const.p+1))@phigk.T@Xgk
                beta_gk[:,k] = temp.ravel() # Maximization w.r.t betagk
                # Maximization w.r.t au sigma_gk :   
                if const.variance_type.lower() == 'common':
                    sk = sum((Xgk-phigk@beta_gk[:,k])**2)
                    s = s+sk
                    sigma_gk = s/sum(sum(tauijk))
                else:
                    temp = phigk@np.array([beta_gk[:,k]]).T
                    sigma_gk[k]= sum((Xgk-temp)**2)/(sum(segment_weights))
                    
            self.param.beta_g[g,:,:] = beta_gk
            self.param.sigma_g[g,:] = list(sigma_gk)
            
            """
            Maximization w.r.t W 
            IRLS : Regression logistique multinomiale pondérée par cluster
            """
            Wg_init = self.param.Wg[g,:,:]
            
            irls.runIRLS(tauijk, phiW[cluster_labels==g,:],  Wg_init)
            
            
            self.param.Wg[g,:,:]=irls.wk;             
            self.param.pi_jgk[g,:,:] = np.matlib.repmat(irls.piik[0:const.m,:],const.n,1); 