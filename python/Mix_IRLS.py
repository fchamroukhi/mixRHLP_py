#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:04:19 2018

@author: bartcus
"""
import utils as utl
import constants as const
import numpy as np

#load lambda;
lmda = 1e-9 # cas du MAP ( a priori gaussien sur W) (penalisation L2)

class IRLS():
    """
     res = IRLS(Winit,M,Tau,verbose) : estime le parametre et les pobas d'un processus logistique temporel
        dans un contexte multi-classes �tant donn�s une partion et une variable 
        explicative, par maximum de vraisembmance (ou MAP)mise en oeuvre par 
        l'algorithme Iteratively Reweighted Least Squares. Le modele est le suivant:
        probas(i,k) = p(zi=k;W)
                    = \pi_{ik}(W)
                    =       exp(wk'vi)
                      -----------------------
                      sum_{l=1}^{K} exp(wl'vi)
                                  
       avec :
            * probas(i,k) est la proba de la classe k au temps t_i :
            i=1,...n et k=1,...,K.
            * vi = [1,ti,ti^2,...,ti^q]^T;
            * q : l'ordre du mod�le logistique
       Le paramtere W=[w1,...,wK] (dans l'algo wK est suppose nul);

 Entrees :

         Winit : parametre initial W(0). dim(Winit) = [(q+1)x(K-1)]
         M : matrice des variables explicatives.  dim(X) = [nx(q+1)]
                            M = [1 t1 t1^2 ... t1^q
                                 1 t2 t2^2 ... t2^q
                                      ..
                                 1 ti ti^2 ... ti^q
                                      ..
                                 1 tn tn^2 ... tn^q]
         Tau : matrice de la partion dure ou floue (ici floue : les pro a 
           posteiori (tik) obtenus par EM). 
         verbose : a mettre � zero si on veut afficher le critere (la
         vraisemblance) aucours des iterations de l'algorithme (par defaut
         verbose = 0)

 Sorties :

          res : structure contenant les resultats. les champs de la
          structure sont:
              wk : le parametre W. matrice de dim [(q+1)x(K-1)]
                  (le dernier etant nul)
              piik :les probabilites
              loglik : la vraisemblance � la convergence de l'algorithme
              LL : vecteur conetant la vraisemblance a chaque iteration de
              l'IRLS.
              reg_irls : log de la proba a priori de W (pour l'injecter
              dans l'EM dans le cas du MAP dans l'IRLS)

  voir article "Improved learning algorithms for mixture of experts in
  multiclass classification" K. Chen, L. Xu & H. Chi. Neural Networks 1999

 Faicel 31 octobre 2008 (mise � jour)
    """
    def __init__(self):
        self.wk = None
        self.LL= None
        self.loglik = None
        self.piik = None
        self.reg_irls = None
        
        
        
    def runIRLS(self, Gamma, Tau, M, Winit = None, trace=False):
        if trace:
            utl.detect_path(const.TraceDir)
            utl.fileGlobalTrace=open(const.TraceDir + "IRLS_Trace{0}.txt".format(const.dataName), "w")
        utl.globalTrace("Start IRLS\n")
        
        
        #n,K = Tau.shape
        n,q = M.shape; #q ici c'est (q+1)
        if Winit is None:
            Winit = np.zeros((q,const.K-1))
        
        I = np.eye(q*(const.K-1));
        
        #Initialisation du IRLS (iter = 0)
        W_old = Winit;
        
        piik_old, loglik_old = utl.modele_logit(W_old,M,Tau,Gamma);
        loglik_old = loglik_old - pow(lmda*(np.linalg.norm(W_old.T.ravel(),2)),2)
        
        iteration = 0;
        converge = False;
        max_iter = 300;
        LL = [];
        
        utl.globalTrace('IRLS : Iteration {0} Log-vraisemblance {1} \n'.format(iteration, loglik_old))
        
        while not converge and  (iteration<max_iter):
            # Hw_old matrice carree de dimensions hx x hx
            hx = q*(const.K-1)
            Hw_old = np.zeros((hx,hx))
            gw_old = np.zeros((q,const.K-1))# todo: verify with matlab this line?
            
            # Gradient :
            for k in range(0,const.K-1):
                gwk = Gamma*np.array([(Tau[:,k] - piik_old[:,k])]).T
                for qq in range(0,q):
                    vq = M[:,qq]
                    gw_old[qq,k] = gwk.T@vq
                    
            gw_old = np.array([np.reshape(gw_old,q*(const.K-1),1)]).T;
            
            # Hessienne
            for k in range(0,const.K-1):
                for ell in range(0, const.K-1):
                    delta_kl=int(k==ell) # kronecker delta 
                    gwk = Gamma*(np.array([piik_old[:,k]]).T*(np.ones((n,1))*delta_kl - np.array([piik_old[:,ell]]).T))
                    Hkl = np.zeros((q,q))
                    for qqa in range(0,q):
                        vqa=np.array([M[:,qqa]]).T
                        for qqb in range(0,q):
                            vqb=np.array([M[:,qqb]]).T  
                            hwk = vqb.T@(gwk*vqa)
                            Hkl[qqa,qqb] = hwk[0,0]
                            
                    
                    Hw_old[k*q : (k+1)*q, ell*q : ell*q+2] = -Hkl
                    
                    
            
            # si a priori gaussien sur W (lambda ~ 1e-9)
            Hw_old = Hw_old + lmda*I;
            gw_old = gw_old - np.array([lmda*W_old.T.ravel()]).T
            # Newton Raphson : W(c+1) = W(c) - H(W(c))^(-1)g(W(c))  
            w = np.array([W_old.T.ravel()]).T - np.linalg.inv(Hw_old)@gw_old ; #[(q+1)x(K-1),1]
            W = np.reshape(w,(q,const.K-1)).T #[(q+1)*(K-1)] 
            
            # mise a jour des probas et de la loglik
            piik, loglik = utl.modele_logit(W, M, Tau ,Gamma)
            
            loglik = loglik - lmda*pow(np.linalg.norm(W.T.ravel(),2),2)
            
            """
            Verifier si Qw1(w^(c+1),w^(c))> Qw1(w^(c),w^(c)) 
            (adaptation) de Newton Raphson : W(c+1) = W(c) - pas*H(W)^(-1)*g(W)
            """
#            pas = 1; # initialisation pas d'adaptation de l'algo Newton raphson
#            alpha = 2;
#            print(loglik)
#            print(loglik_old)
#            while (loglik < loglik_old):
#                pas = pas/alpha; # pas d'adaptation de l'algo Newton raphson
#                #recalcul du parametre W et de la loglik
#                w = np.array([W_old.T.ravel()]).T - pas*np.linalg.inv(Hw_old)@gw_old ;
#                W = np.reshape(w,(q,const.K-1)).T
#                # mise a jour des probas et de la loglik
#                print('Start model logit')
#                piik, loglik = utl.modele_logit(W, M, Tau ,Gamma)
#                print('end model logit')
#                loglik = loglik - lmda*pow(np.linalg.norm(W.T.ravel(),2),2)
                
            converge1 = abs((loglik-loglik_old)/loglik_old) <= 1e-7
            converge2 = abs(loglik-loglik_old) <= 1e-6
            
            converge = converge1 | converge2
            
            piik_old = piik;
            W_old = W;
            iteration += 1;
            LL.append(loglik_old)
            loglik_old = loglik
            
            utl.globalTrace('IRLS : Iteration {0} Log-vraisemblance {1} \n'.format(iteration, loglik_old))
        utl.globalTrace('Fin IRLS \n')
        
        if converge:
            utl.globalTrace('IRLS : convergence  OK ; nbre d''iterations : {0}\n'.format(iteration))
        else:
            utl.globalTrace('\nIRLS : pas de convergence (augmenter le nombre d''iterations > {0}) \n'.format(iteration))
            
            
        self.wk = W;
        self.LL= LL;
        self.loglik = loglik;
        self.piik = piik;    
        if lmda!=0: #pour l'injection de l'a priori dans le calcul de la  loglik de l'EM dans le cas du MAP
            self.reg_irls = - lmda*pow(np.linalg.norm(W.T.ravel(),2),2)
        else:
            self.reg_irls = 0
            
        
        if trace:
            utl.fileGlobalTrace.close()
            utl.fileGlobalTrace = None  
            
            
def testIRLS():
    import scipy.io
    mat = scipy.io.loadmat('data/IRLStest.mat')

    Winit = np.zeros((2,2))
    Gamma = mat['cluster_weights']
    M = mat['phiW']
    Tau = mat['tauijk']
    irls = IRLS()
    irls.runIRLS(Gamma, Tau, M, Winit)
    del mat
    return irls
    
#irls = testIRLS()   
    