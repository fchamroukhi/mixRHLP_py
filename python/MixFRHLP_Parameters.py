#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:52:18 2018

@author: bartcus
"""

from sklearn.cluster import KMeans
import numpy as np
import constants as const
import utils as utl


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
        self.Wg = np.zeros([const.G, const.q+1, const.K-1])
        self.beta_g = np.NaN * np.empty([const.G, const.p +1, const.K])
        
        if const.variance_type.lower() == 'common':
            self.sigma_g = np.NaN * np.empty([const.G, 1])
        else:
            self.sigma_g = np.NaN * np.empty([const.G, const.K])
            
        self.pi_jgk = np.NaN * np.empty([const.G, const.m*const.n, const.K])
        self.alpha_g = np.NaN * np.empty(const.G)
        
    def initialize_MixFRHLP_EM(self, phiBeta, phiW, try_EM):
        # 1. Initialization of cluster weights
        self.alpha_g=1/const.G*np.ones(const.G)
        
        # 2. Initialization of the model parameters for each cluster: W (pi_jgk), betak and sigmak    
        #self.Wg, self.pi_jgk = 
        self.__initHlp(phiW, try_EM)
        
        # 3. Initialization of betagk and sigmagk
        if const.init_kmeans:
            kmeans = KMeans(n_clusters = const.G, init = 'k-means++', max_iter = 400, n_init = 20, random_state = 0)
            klas = kmeans.fit_predict(const.data)
            
            #klas = [np.zeros(10),[1]*np.ones(10), [2]*np.ones(10)]
            #klas = np.hstack(klas)
            
            for g in range(0,const.G):
                Xg = const.data[klas==g ,:]; #if kmeans  
                betak, sigma = self.__initRegressionParam(Xg, phiBeta, try_EM)
                
                self.beta_g[g,:,:] = betak;
                if const.variance_type.lower() == 'common':
                    self.sigma_g[g] = sigma;
                else:
                    self.sigma_g[g,:] = sigma;
        else:
            print('todo: line 41 matlab initialize_MixFRHLP_EM')
            raise RuntimeError('todo: line 41 matlab initialize_MixFRHLP_EM')
        
        
    def __initRegressionParam(self, Xg, phiBeta, try_EM):
        """
        aim: initialize the Regresssion model with Hidden Logistic Process
        requires:
            data - the data set
            K - the number of regimes
            phi
            variance_type - variance type
            try_EM - em try
        ensures:
            sigma
            betak
        """
        n, m = Xg.shape
        if try_EM == 1:
            # Decoupage de l'echantillon (signal) en K segments
            zi = round(m/const.K) - 1
            #todo ameliorate code for initialization of sigma and betak
            sigma=[]
            betak_list = []
            for k in range(1, const.K+1):
                i = (k-1)*zi;
                j = k*zi;
                Xij = Xg[:,i:j];
                Xij = np.reshape(Xij,(np.prod(Xij.shape), 1))
                phi_ij = phiBeta[i:j,:];
                Phi_ij = np.matlib.repmat(phi_ij, n, 1);
                bk = np.linalg.inv(Phi_ij.T@Phi_ij)@Phi_ij.T@Xij;
                #para.betak(:,k) = bk;
                betak_list.append(bk)
                if const.variance_type.lower() == 'common':
                    sigma = np.var(Xij)
                else:
                    mk = j-i
                    z = Xij-Phi_ij@bk;
                    
                    sk = z.T@z/(n*mk); 
                    sigma.append(sk[0][0])
            #remake betak
            betak = np.hstack(betak_list)
        else:
            #random initialization
            Lmin= round(m/(const.K+1)) #nbr pts min dans un segments
            tk_init = [0] * (const.K+1)
            tk_init[0]=-1
            K_1=const.K;
            #todo: verify indexes ???
            for k in range(1,const.K):
                K_1 = K_1-1
                temp = np.arange(tk_init[k-1]+Lmin,m-K_1*Lmin)
                ind = np.random.permutation(len(temp))
                tk_init[k]= temp[ind[0]];
                
            tk_init[const.K] = m-1; 
            
            sigma=[]
            betak_list = []
            for k in range(0, const.K):
                i = tk_init[k]+1;
                j = tk_init[k+1];
                Xij = Xg[:,i:j];
                Xij = np.reshape(Xij,(np.prod(Xij.shape), 1))
                phi_ij = phiBeta[i:j,:];
                Phi_ij = np.matlib.repmat(phi_ij, n, 1);
                bk = np.linalg.inv(Phi_ij.T@Phi_ij)@Phi_ij.T@Xij;
                betak_list.append(bk)
                if const.variance_type.lower() == 'common':
                    sigma = np.var(Xij)
                else:
                    mk = j-i
                    z = Xij-Phi_ij@bk;
                    #todo: verify if sk is always one value and not a matrix
                    sk = z.T@z/(n*mk); 
                    sigma.append(sk[0][0])
            #remake betak
            betak = np.hstack(betak_list)
        #print(betak)
        #print(sigma)
        #wait = input('PLEASE PRESS ENTER')
        return betak, sigma
                
    def __initHlp(self, phiW, try_EM):
        """
            Initialize the Hidden Logistic Process
        """
        # 1. Initialisation de W (pi_jgk)
        nm, q1 = phiW.shape;
        if  try_EM ==1:
            for g in range(0,const.G):
                self.pi_jgk[g,:,:] = utl.modele_logit(self.Wg[g,:,:],phiW)[0];
        else:
            for g in range(0,const.G):
                self.Wg[g,:,:] = np.random.rand(const.q+1,const.K-1); #initialisation aléatoire du vercteur param�tre du IRLS
                self.pi_jgk[g,:,:] = utl.modele_logit(self.Wg[g,:,:],phiW)[0];
                
def main_initialize():
    # 1. Construction des matrices de regression
    x = np.linspace(0,1,const.m) # ou rentrer le vecteur de covariables des courbes
    # 2. pour 1 courbe
    phiBeta, phiW = utl.designmatrix_FRHLP(x, const.p, const.q);
    #pour les n courbes (regularly sampled)
    phiBeta = np.matlib.repmat(phiBeta, const.n, 1);
    phiW = np.matlib.repmat(phiW, const.n, 1);
    try_EM = 1
    
    param = MixParam()
    param.initialize_MixFRHLP_EM(phiBeta, phiW, try_EM)
    return param
    
#param = main_initialize()
    
    
