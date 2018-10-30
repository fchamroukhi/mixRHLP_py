#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:00:40 2018

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
            self.sigma_g = np.NaN * np.empty([const.G])
        else:
            self.sigma_g = np.NaN * np.empty([const.G, const.K])
            
        self.pi_jgk = np.NaN * np.empty([const.m*const.n, const.K, const.G])
        self.alpha_g = np.NaN * np.empty(const.G)
        
    def initialize_MixFRHLP_EM(self, phiBeta, phiW, try_EM):
        # 1. Initialization of cluster weights
        self.alpha_g=1/const.G*np.ones(const.G)
        
        # 2. Initialization of the model parameters for each cluster: W (pi_jgk), betak and sigmak    
        self.Wg, self.pi_jgk = self.__initHlp(phiW, try_EM)
        
        # 3. Initialization of betagk and sigmagk
        if const.init_kmeans:
            kmeans = KMeans(n_clusters = const.G, init = 'k-means++', max_iter = 400, n_init = 20, random_state = 0)
            klas = kmeans.fit_predict(const.data)
            for g in range(0,const.G):
                Xg = const.data[klas==g ,:]; #if kmeans  
                betak, sigma = self.__initRegressionParam(phiBeta, try_EM)
                
                self.beta_g[g,:,:] = betak;
                if const.variance_type.lower() == 'common':
                    self.sigma_g[g] = sigma;
                else:
                    self.sigma_g[g,:] = sigma;
        else:
            print('todo: line 41 matlab initialize_MixFRHLP_EM')
        
        
        
    def __initRegressionParam(self, phiBeta, try_EM):
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
        if try_EM == 1:
            # Decoupage de l'echantillon (signal) en K segments
            zi = round(const.m/const.K) - 1
            #todo ameliorate code for initialization of sigma and betak
            sigma=[]
            betak_list = []
            for k in range(1, const.K+1):
                i = (k-1)*zi;
                j = k*zi;
                Xij = const.data[:,i:j];
                Xij = np.reshape(Xij.T,(np.prod(Xij.shape), 1))
                phi_ij = phiBeta[i:j,:];
                Phi_ij = np.matlib.repmat(phi_ij, const.n, 1);
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
            Lmin= round(const.m/(const.K+1));#nbr pts min dans un segments
            tk_init = [0] * (const.K+1)
            K_1=const.K;
            #todo: verify indexes ???
            for k in range(0,const.K-1):
                K_1 = K_1-1;
                temp = np.arange(tk_init[k]+Lmin,const.m-K_1*Lmin+1)
                ind = np.random.permutation(len(temp))
                tk_init[k+1]= temp[ind[0]];
                
            tk_init[const.K] = const.m; 
            
            sigma=[]
            betak_list = []
            for k in range(0, const.K-1):
                i = tk_init[k];
                j = tk_init[k+1];
                Xij = const.data[:,i:j];
                Xij = np.reshape(Xij.T,(np.prod(Xij.shape), 1))
                phi_ij = phiBeta[i:j,:];
                Phi_ij = np.matlib.repmat(phi_ij, const.n, 1);
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
            
        return betak, sigma
                
    def __initHlp(self, klas, phiW, try_EM):
        """
            Initialize the Hidden Logistic Process
        """
        # 1. Initialisation de W (pi_jgk)
        nm, q1 = phiW.shape;

        if  try_EM ==1
            for g=1:G
                self.pi_jgk[g,:,:] = utl.modele_logit(self.Wg[g,:,:],phiW);
        else
            for g=1:G
                self.Wg[g,:,:] = np.random.rand(q+1,K-1); #initialisation aléatoire du vercteur param�tre du IRLS
                self.pi_jgk[g,:,:] = utl.modele_logit(self.Wg[g,:,:],phiW);
        
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
            self.param = MixParam()
            self.param.initialize_MixFRHLP_EM(phiBeta, phiW, try_EM)
            
            iteration = 0; 
            converge = False
            prev_loglik = -np.Inf
            
            #EM
            self.tau_ijgk = np.zeros(G, const.n*const.m, K) # segments post prob  
            log_tau_ijgk = np.zeros(G, const.n*const.m, K)
            
            log_fg_xij = np.zeros(n,G); 
            log_alphag_fg_xij = np.zeros(n,G); 
            
            
            while not(converge) and (iteration<= max_iter_EM):
                """
                E-Step
                """
                
                """
                M-Step
                """
            
            
            cpu_time = time.time()-start_time
            cputime_total.append(cpu_time)
            
            
        utl.globalTrace("End EM\n")
        if trace:
            utl.fileGlobalTrace.close()
            utl.fileGlobalTrace = None 
            
    
#solution =  MixFRHLP_EM(data); 
