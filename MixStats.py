#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:47:01 2018

@author: Faïcel Chamroukhi & Bartcus Marius
"""
import numpy as np
import enums
import sys
import utils as utl
import matplotlib.pyplot as plt

class MixStats():
    def __init__(self, mixModel, options):
        self.h_ig = np.NaN*np.empty([mixModel.n, mixModel.G])
        self.c_ig = np.NaN*np.empty([mixModel.n, mixModel.G])
        self.klas = np.NaN*np.empty([mixModel.n, 1])
        self.Ex_g = np.NaN*np.empty([mixModel.m, mixModel.G])
        
        self.loglik = -np.Inf
        self.comp_loglik = -np.Inf
        self.stored_loglik = np.NaN*np.empty([options.max_iter, options.n_tries])
        self.stored_com_loglik = np.NaN*np.empty([options.max_iter, options.n_tries])
        
        self.BIC = -np.Inf
        self.AIC = -np.Inf
        self.ICL = -np.Inf
        
        self.cpu_time = -np.Inf
        
        self.log_fg_xij = np.zeros((mixModel.n, mixModel.G))
        self.log_alphag_fg_xij = np.zeros((mixModel.n, mixModel.G))
        self.polynomials = np.NaN * np.empty([mixModel.G, mixModel.m, mixModel.K])
        self.weighted_polynomials = np.NaN * np.empty([mixModel.G, mixModel.m, mixModel.K])
        self.tau_ijgk = np.NaN*np.empty([mixModel.G, mixModel.n * mixModel.m, mixModel.K])
        self.log_tau_ijgk = np.NaN*np.empty([mixModel.G, mixModel.n * mixModel.m, mixModel.K])
    
    def showDataClusterSegmentation(self, mixModel, mixParamSolution):
        plt.figure(1, figsize=(10,8))
        plt.plot(self.stored_loglik)
        plt.xlabel('iteration')
        plt.ylabel('objective function')
        #plt.savefig('figures/stored_loglik.png')
        
        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}
    
        plt.matplotlib.rc('font', **font)
        t = np.arange(mixModel.m)
        G = len(mixParamSolution.alpha_g);
        colors = ['r','b','g','m','c','k','y']
        colors_cluster_means = [[0.8, 0, 0],[0, 0, 0.8],[0, 0.8, 0],'m','c','k','y']
        
        
        plt.figure(2, figsize=(10,8))        
        plt.plot(t,mixModel.X.T);    
        plt.title('Original times series')
        plt.xlabel('Time')
        plt.ylabel('y')        
        
        plt.figure(3, figsize=(10,8))
        
        for g in range(0,G):
            cluster_g = mixModel.X[self.klas==g ,:];
            plt.plot(t,cluster_g.T,colors[g],linewidth=0.1);    
            plt.plot(t, self.Ex_g[:,g], colors_cluster_means[g],linewidth=3)
            
        plt.title('Clustered and segmented times series')
        plt.xlabel('Time')
        plt.ylabel('y')
        plt.xlim(0, mixModel.m-1)
        #plt.savefig('figures/data_clustering.png')
        
        
        for g in range(0,G):
            plt.figure(g+4,figsize=(10,8))
            plt.subplot(2, 1, 1)
            cluster_g = mixModel.X[self.klas==g ,:]
            plt.plot(t,cluster_g.T,colors[g])
            plt.plot(t,self.polynomials[g,:,:],'k--',linewidth=1)
            plt.plot(t,self.Ex_g[:,g],colors_cluster_means[g],linewidth=3)
            plt.title('Cluster {0}'.format(g+1))
            plt.ylabel('y');
            plt.xlim([0, mixModel.m-1])
            
            plt.subplot(2, 1, 2)
            plt.plot(t, mixParamSolution.pi_jgk[g,0:mixModel.m,:],linewidth=2);
            plt.ylabel('Logistic proportions')
            plt.xlabel('Time')
            ax = plt.gca()
            ax.set_yticklabels(np.linspace(0,1,6))
            plt.xlim([0, mixModel.m-1])
            
            #plt.savefig('figures/cluster{0}.png'.format(g))
        plt.show()
        
    def computeStats(self, mixModel, mixParam, phi, cpu_time_all):
        for g in range(0,mixModel.G):
            self.polynomials[g,:,:] = phi.XBeta[0:mixModel.m,:]@mixParam.beta_g[g,:,:]
            
            self.weighted_polynomials[g,:,:] = mixParam.pi_jgk[g,:,:]*self.polynomials[g,:,:]
            self.Ex_g[:,g] = self.weighted_polynomials[g,:,:].sum(axis=1); 
        
        self.Ex_g = self.Ex_g[0:mixModel.m,:] 
        self.cpu_time = np.mean(cpu_time_all)
        
        Psi = np.array([mixParam.alpha_g.T.ravel(), mixParam.Wg.T.ravel(), mixParam.beta_g.T.ravel(), mixParam.sigma_g.T.ravel()])
        nu = len(Psi);
        # BIC AIC et ICL*
        self.BIC = self.loglik - (nu*np.log(mixModel.n)/2) #n*m/2!
        self.AIC = self.loglik - nu
        # ICL*             
        # Compute the comp-log-lik 
        cig_log_alphag_fg_xij = (self.c_ig)*(self.log_alphag_fg_xij);
        self.comp_loglik = sum(cig_log_alphag_fg_xij.sum(axis=1)) 

        self.ICL = self.comp_loglik - nu*np.log(mixModel.n)/2 #n*m/2!
        
        self.klas = self.klas.reshape(mixModel.n)
     # Bayes allocation rule   
    def MAP(self):
        """
        % calculate a partition by applying the Maximum A Posteriori Bayes
        % allocation rule
        %
        %
        % Inputs : 
        %   PostProbs, a matrix of dimensions [n x K] of the posterior
        %  probabilities of a given sample of n observations arizing from K groups
        %
        % Outputs:
        %   klas: a vector of n class labels (z_1, ...z_n) where z_i =k \in {1,...K}
        %       klas(i) = arg   max (PostProbs(i,k)) , for all i=1,...,n
        %                     1<=k<=K
        %               = arg   max  p(zi=k|xi;theta)
        %                     1<=k<=K
        %               = arg   max  p(zi=k;theta)p(xi|zi=k;theta)/sum{l=1}^{K}p(zi=l;theta) p(xi|zi=l;theta)
        %                     1<=k<=K
        %
        %
        %       Z : Hard partition data matrix [nxK] with binary elements Zik such
        %       that z_ik =1 iff z_i = k
        %
        """
        N, K = self.h_ig.shape
        
        ikmax = np.argmax(self.h_ig,axis=1)
        ikmax = np.reshape(ikmax,(ikmax.size,1))
        self.c_ig = (ikmax@np.ones((1,K))) == (np.ones((N,1))@np.array([range(0,K)]));
        self.klas = np.ones((N,1))
        for k in range(0,K):
            self.klas[self.c_ig[:,k]==1]=k
            
        # assignement step
    
    def CStep(self, reg_irls):
        # cluster posterior probabilities p(c_i=g|X)
        self.h_ig = np.exp(utl.log_normalize(self.log_alphag_fg_xij))
        
        self.MAP(); # c_ig the hard partition of the curves 
 
        #Compute the optimized criterion  
        cig_log_alphag_fg_xij = self.c_ig*self.log_alphag_fg_xij;
        self.comp_loglik = sum(cig_log_alphag_fg_xij.sum(axis=1)) +  reg_irls
        
    def EStep(self, mixModel, mixParam, phi, variance_type):
        """
        E-step
        """
        for g in range(0,mixModel.G):
            alpha_g = mixParam.alpha_g[g]
            beta_g = mixParam.beta_g[g,:,:]
            #Wg = self.param.Wg[g,:,:]
            pi_jgk = mixParam.pi_jgk[g,:,:]
            
            log_pijgk_fgk_xij = np.zeros((mixModel.n*mixModel.m, mixModel.K))
            for k in range(0,mixModel.K):
                beta_gk = beta_g[:,k]
                if variance_type == enums.variance_types.common :
                    sgk = mixParam.sigma_g[g]
                else:
                    #?
                    sgk = mixParam.sigma_g[g,k]
                
                temp = phi.XBeta@beta_gk
                temp = temp.reshape((len(temp), 1))
                z=((mixModel.XR-temp)**2)/sgk;
                #print(sgk)
                temp = np.array([np.log(pi_jgk[:,k]) - 0.5*(np.log(2*np.pi) + np.log(sgk))]).T - 0.5*z
                log_pijgk_fgk_xij[:,k] = temp.T #pdf cond à c_i = g et z_i = k de xij
                
                
            log_pijgk_fgk_xij = np.minimum(log_pijgk_fgk_xij,np.log(sys.float_info.max))
            log_pijgk_fgk_xij = np.maximum(log_pijgk_fgk_xij,np.log(sys.float_info.min))
            
            pijgk_fgk_xij = np.exp(log_pijgk_fgk_xij);
            sumk_pijgk_fgk_xij = np.array([pijgk_fgk_xij.sum(axis = 1)]).T # sum over k
            log_sumk_pijgk_fgk_xij  = np.log(sumk_pijgk_fgk_xij) #[nxm x 1]
            
            self.log_tau_ijgk[g,:,:] = log_pijgk_fgk_xij - log_sumk_pijgk_fgk_xij @ np.ones((1,mixModel.K))
            self.tau_ijgk[g,:,:] = np.exp(utl.log_normalize(self.log_tau_ijgk[g,:,:]))
            
            temp = np.reshape(log_sumk_pijgk_fgk_xij.T,(mixModel.n, mixModel.m))
            self.log_fg_xij[:,g] = temp.sum(axis = 1) #[n x 1]:  sum over j=1,...,m: fg_xij = prod_j sum_k pi_{jgk} N(x_{ij},mu_{gk},s_{gk))
            self.log_alphag_fg_xij[:,g] = np.log(alpha_g) + self.log_fg_xij[:,g] # [nxg] 
            
        self.log_alphag_fg_xij = np.minimum(self.log_alphag_fg_xij,np.log(sys.float_info.max))
        self.log_alphag_fg_xij = np.maximum(self.log_alphag_fg_xij,np.log(sys.float_info.min))
        
        # cluster posterior probabilities p(c_i=g|X)
        self.h_ig = np.exp(utl.log_normalize(self.log_alphag_fg_xij))
        # log-likelihood
        temp = np.exp(self.log_alphag_fg_xij)
        self.loglik = sum(np.log(temp.sum(axis = 1)))

        