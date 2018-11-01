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
    def __init__(self, ):
        self.wk = None
        self.LL= None
        self.loglik = None
        self.piik = None
        self.reg_irls = None
        
        
        
    def runIRLS(self, Gamma,Tau,M, Winit = None, trace=False):
        if trace:
            utl.detect_path(const.TraceDir)
            utl.fileGlobalTrace=open(const.TraceDir + "IRLS_Trace{0}.txt".format(const.dataName), "w")
        utl.globalTrace("Start IRLS\n")
        
        
        #n,K = Tau.shape
        n,q = M.shape; #q ici c'est (q+1)
        if Winit == None:
            Winit = np.zeros((q,const.K-1))
        else:
            Winit = Winit
        
        I = np.eye(q*(const.K-1));
        
        #Initialisation du IRLS (iter = 0)
        W_old = Winit;
        
        piik_old, loglik_old = utl.modele_logit(W_old,M,Tau,Gamma);
        loglik_old = loglik_old - pow(lmda*(np.linalg.norm(W_old[:],2)),2)
        
        iteration = 0;
        converge = False;
        max_iter = 300;
        LL = [];
        
        utl.globalTrace('IRLS : Iteration {0} Log-vraisemblance {1} \n'.format(iteration, loglik_old))
        
        while not converge and  (iteration<max_iter):
            # Hw_old matrice carree de dimensions hx x hx
        
        if lmda!=0: #pour l'injection de l'a priori dans le calcul de la  loglik de l'EM dans le cas du MAP
            self.reg_irls = - pow(lmda*(np.linalg.norm(W[:],2)),2)
        else:
            self.reg_irls = 0