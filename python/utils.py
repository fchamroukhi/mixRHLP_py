#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:29:21 2018

@author: bartcus
"""
import numpy as np
import os
from sklearn.preprocessing import normalize
import default_constants as defConst

fileGlobalTrace=None

def globalTrace(message):
    """
    aim: prints a message in a file
    input: 
        message
    """
    if not fileGlobalTrace is None:
        fileGlobalTrace.write(message)
        fileGlobalTrace.flush()
        
def detect_path(pathname):
    """
    requires: 
        a path name
    ensures:
        creqtes the path if it does not exist
    """
    if not os.path.exists(pathname):
        os.makedirs(pathname, exist_ok=True)
        

def modele_logit(W,M,Y=None, Gamma=None):
    """
     [probas, loglik] = modele_logit(W,X,Y)M try: 1
    
     calcule les pobabilites selon un le modele logistique suivant :
    
     probas(i,k) = p(zi=k;W)= \pi_{ik}(W) 
                            =          exp(wk'vi)
                              ----------------------------
                             1 + sum_{l=1}^{K-1} exp(wl'vi)
     for all i=1,...,n et k=1...K (dans un contexte temporel on parle d'un
     processus logistique)
    
     Entrees :
    
             1. W : parametre du modele logistique ,Matrice de dimensions
             [(q+1)x(K-1)]des vecteurs parametre wk. W = [w1 .. wk..w(K-1)] 
             avec les wk sont des vecteurs colonnes de dim [(q+1)x1], le dernier 
             est suppose nul (sum_{k=1}^K \pi_{ik} = 1 -> \pi{iK} =
             1-sum_{l=1}^{K-1} \pi{il}. vi : vecteur colonne de dimension [(q+1)x1] 
             qui est la variable explicative (ici le temps): vi = [1;ti;ti^2;...;ti^q];
             2. M : Matrice de dimensions [nx(q+1)] des variables explicatives. 
                M = transpose([v1... vi ....vn]) 
                  = [1 t1 t1^2 ... t1^q
                     1 t2 t2^2 ... t2^q
                           ..
                     1 ti ti^2 ... ti^q
                           ..
                     1 tn tn^2 ... tn^q]
               q : ordre de regression logistique
               n : nombre d'observations
            3. Y Matrice de la partition floue (les probas a posteriori tik)
               tik = p(zi=k|xi;theta^m); Y de dimensions [nxK] avec K le nombre de classes
     Sorties : 
    
            1. probas : Matrice de dim [nxK] des probabilites p(zi=k;W) de la vaiable zi
              (i=1,...,n)
            2. loglik : logvraisemblance du parametre W du modele logistique
               loglik = Q1(W) = Esperance(l(W;Z)|X;theta^m) = E(p(Z;W)|X;theta^m) 
                      = logsum_{i=1}^{n} sum_{k=1}^{K} tik log p(zi=k;W)
       
     Cette fonction peut egalement �tre utilis�e pour calculer seulement les 
     probas de la fa�oc suivante : probas = modele_logit(W,M)
    """
    #todo: verify this code when Y != none
    if Y is not None:
        n1, K = Y.shape
        
        if Gamma is not None:
            Gamma = Gamma*np.ones((1,K))
            
        n2, q = M.shape # ici q c'est q+1
        if n1==n2:
            n=n1;
        else:
            raise ValueError(' W et Y doivent avoir le meme nombre de ligne')
    else:
        n,q=M.shape
        
    if Y is not None:
        #todo: finish this code
        if np.size(W,1) == (K-1): # pas de vecteur nul dans W donc l'ajouter
            wK=np.zeros((q,1));
            W = np.concatenate((W, wK), axis=1) # ajout du veteur nul pour le calcul des probas
        else:
            raise ValueError(' W et Y doivent avoir le meme nombre de ligne')
    else:
        wK=np.zeros((q,1));
        W = np.concatenate((W, wK), axis=1) # ajout du veteur nul pour le calcul des probas
        q,K = W.shape;
    
    
    MW = M@W; # multiplication matricielle
    maxm = MW.max(1).reshape((len(MW.max(1)), 1))
    MW = MW - maxm @ np.ones((1,K)); #normalisation
    
    expMW = np.exp(MW)
    
    frc = expMW[:,0:K].sum(axis = 1)
    frc = np.reshape(frc, (frc.size,1))
    frc = frc@np.ones((1,K))
    
    probas = expMW/frc;
    
    if Y is not None:
        if Gamma is None:
            temp = Y*np.log(expMW.sum(axis=1)*np.ones((1,K)))
            temp = (Y*MW) - temp
            loglik = sum(temp.sum(axis=1))
        else:
            temp = (Gamma*Y)*np.log(np.array([expMW.sum(axis=1)]).T*np.ones((1,K)))
            temp = (Gamma*(Y*MW)) - temp
            loglik = sum(temp.sum(axis=1))
            
        
        if np.isnan(loglik):
            MW=M@W;
            minm = -745.1;
            MW = np.maximum(MW, minm);
            maxm = 709.78;
            MW= np.minimum(MW,maxm);
            expMW = np.exp(MW);
            
            if Gamma is None:
                temp=Y*np.log(expMW.sum(axis=1)*np.ones((1,K))+defConst.eps)
                temp=(Y*MW) - temp
                loglik = sum(temp.sum(axis=1))
            else:
                temp=(Gamma*Y)*np.log(expMW.sum(axis=1)*np.ones((1,K))+defConst.eps)
                temp=((Gamma*Y)*MW) - temp
                loglik = sum(temp.sum(axis=1))
                
        if np.isnan(loglik):
            raise ValueError('Probleme loglik IRLS NaN (!!!)')
    else:
        loglik = []
            
    return probas, loglik
    
def designmatrix_FRHLP(x,p,q=None):
    """
    requires:
        x - data
        p - dimension de beta (ordre de reg polynomiale)
        q (Optional) - dimension de w (ordre de reg logistique)
    ensures:
        creates the parameters phiBeta and phiW
    """
    if x.shape[0] == 1:
        x=x.T; # en vecteur
    
    order_max = p    
    if q!=None:
        order_max = max(p,q)
        
    phi=np.NaN * np.empty([len(x), order_max+1])
    for ordr in range(order_max+1):
        phi[:,ordr] = x**ordr # phi2w = [1 t t.^2 t.^3 t.^p;......;...]
    #todo: verify        
    phiBeta = phi[:,0:p+1]; # Matrice de regresseurs pour Beta

    phiW =[];
    if q!=None:
        phiW = phi[:,0:q+1]; # matrice de regresseurs pour w
    
    return phiBeta, phiW
            


def MAP(post_probas):
    """
     calcule une partition d'un echantillon par la regle du Maximum A Posteriori à partir des
    
     probabilites a posteriori 
    
     Entrees : post_probas , Matrice de dimensions [n x K] des probabibiltes a
     posteriori (matrice de la partition floue)
    
           n : taille de l'echantillon
    
           K : nombres de classes
    
           klas(i) = arg   max (post_probas(i,k)) , for all i=1,...,n
                         1<=k<=K
                   = arg   max  p(zi=k|xi;theta)
                         1<=k<=K
                   = arg   max  p(zi=k;theta)p(xi|zi=k;theta)/sum{l=1}^{K}p(zi=l;theta) p(xi|zi=l;theta)
                         1<=k<=K
    
     Sorties : classes : vecteur collones contenant les classe (1:K)
    
           Z : Matrice de dimension [nxK] de la partition dure : ses elements sont zik, avec zik=1 si xi
           appartient à la classe k (au sens du MAP) et zero sinon.
    
    """
    N, K = post_probas.shape
    
    ikmax = post_probas.max(1)
    ikmax = np.reshape(ikmax,(ikmax.size,1))
    partition_MAP = (ikmax@np.ones((1,K))) == (np.ones((N,1))@np.array([range(0,K)]));
    klas = np.ones((N,1))
    for k in range(0,K):
        klas[partition_MAP[:,k]==1]=k
    Z=partition_MAP
    return klas, Z

"""
    ########################################
    start code for normalization of the data
    ########################################
"""
def log_normalize(matrix):
    # compute x - repmat(log(sum(exp(x),2)),1,size(x,2)) in a robust way
    n, d = matrix.shape
    a = np.array([matrix.max(axis = 1)])
    a=a.T
    temp = np.matlib.repmat(a,1,d)
    temp = np.exp( matrix - temp )
    temp = np.array([temp.sum(axis = 1)])
    temp=temp.T
    return matrix - np.matlib.repmat(a + np.log( temp ) ,1,d)

def normalize_matrix(matrix):
    """
        Scikit-learn normalize function that lets you apply various normalizations. 
        The "make it sum to 1" is the L1 norm
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
    """
    normed_matrix = normalize(matrix, axis=1, norm='l1')  
    return normed_matrix


def test_norm():
    matrix = np.arange(0,27,3).reshape(3,3).astype(np.float64)
    #array([[  0.,   3.,   6.],
    #   [  9.,  12.,  15.],
    #   [ 18.,  21.,  24.]])
    print(normalize_matrix(matrix))
    #[[ 0.          0.33333333  0.66666667]
    #[ 0.25        0.33333333  0.41666667]
    #[ 0.28571429  0.33333333  0.38095238]]
    
#test_norm()
    


    
    
