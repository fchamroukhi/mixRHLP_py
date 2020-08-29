#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:29:21 2018

@author: Faïcel Chamroukhi & Bartcus Marius
"""
import numpy as np
import os
#from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

fileGlobalTrace=None
eps = np.spacing(1)

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
         % calculates the pobabilities according to multinomial logistic model:
         %
         % probs(i,k) = p(zi=k;W)= \pi_{ik}(W)
         %                                  exp(wk'vi)
         %                        =  ----------------------------
         %                          1 + sum_{l=1}^{K-1} exp(wl'vi)
         % for i=1,...,n and k=1...K
         %
         % Inputs :
         %
         %         1. W : parametre du modele logistique ,Matrice de dimensions
         %         [(q+1)x(K-1)]des vecteurs parametre wk. W = [w1 .. wk..w(K-1)]
         %         avec les wk sont des vecteurs colonnes de dim [(q+1)x1], le dernier
         %         est suppose nul (sum_{k=1}^K \pi_{ik} = 1 -> \pi{iK} =
         %         1-sum_{l=1}^{K-1} \pi{il}. vi : vecteur colonne de dimension [(q+1)x1]
         %         qui est la variable explicative (ici le temps): vi = [1;ti;ti^2;...;ti^q];
         %         2. M : Matrice de dimensions [nx(q+1)] des variables explicatives.
         %            M = transpose([v1... vi ....vn])
         %              = [1 t1 t1^2 ... t1^q
         %                 1 t2 t2^2 ... t2^q
         %                       ..
         %                 1 ti ti^2 ... ti^q
         %                       ..
         %                 1 tn tn^2 ... tn^q]
         %           q : ordre de regression logistique
         %           n : nombre d'observations
         %        3. Y Matrice de la partition floue (les probs a posteriori tik)
         %           tik = p(zi=k|xi;theta^m); Y de dimensions [nxK] avec K le nombre de classes
         % Sorties :
         %
         %        1. probs : Matrice de dim [nxK] des probabilites p(zi=k;W) de la vaiable zi
         %          (i=1,...,n)
         %        2. loglik : logvraisemblance du parametre W du modele logistique
         %           loglik = Q1(W) = E(l(W;Z)|X;theta^m) = E(p(Z;W)|X;theta^m)
         %                  = logsum_{i=1}^{n} sum_{k=1}^{K} tik log p(zi=k;W)
         %
         % Cette fonction peut egalement ?tre utilis?e pour calculer seulement les
         % probs de la fa?oc suivante : probs = modele_logit(W,X)
         %
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
    maxm = MW.max(1)
    maxm = maxm.reshape((len(maxm), 1))
    MW = MW - maxm @ np.ones((1,K)); #normalisation
    
    expMW = np.exp(MW)
    
    frc = expMW[:,0:K].sum(axis = 1)
    frc = np.reshape(frc, (frc.size,1))
    frc = frc@np.ones((1,K))
    
    probas = expMW/frc;
    
    if Y is not None:
        if Gamma is None:
            temp = expMW.sum(axis=1)
            temp = np.reshape(temp,(len(temp),1))
            temp = Y*np.log(temp@np.ones((1,K)))
            temp = (Y*MW) - temp
            loglik = sum(temp.sum(axis=1))
        else:
            temp = (Gamma*Y)*np.log(np.array([expMW.sum(axis=1)]).T@np.ones((1,K)))
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
                temp=Y*np.log(expMW.sum(axis=1)@np.ones((1,K))+eps)
                temp=(Y*MW) - temp
                loglik = sum(temp.sum(axis=1))
            else:
                temp=(Gamma*Y)*np.log(np.array([expMW.sum(axis=1)]).T@np.ones((1,K))+eps)
                temp=((Gamma*Y)*MW) - temp
                loglik = sum(temp.sum(axis=1))
                
        if np.isnan(loglik):
            raise ValueError('Probleme loglik IRLS NaN (!!!)')
    else:
        loglik = []
            
    return probas, loglik
    
            


def IRLS(Tau, M, Winit = None, Gamma=None, trace=False):
    """
        % res = IRLS_MixFRHLP(X, Tau, Gamma, Winit, verbose) : an efficient Iteratively Reweighted Least-Squares (IRLS) algorithm for estimating
        % the parameters of a multinomial logistic regression model given the
        % "predictors" X and a partition (hard or smooth) Tau into K>=2 segments,
        % and a cluster weights Gamma (hard or smooth)
        % 
        %
        % Inputs :
        %
        %         X : desgin matrix for the logistic weights.  dim(X) = [nx(q+1)]
        %                            X = [1 t1 t1^2 ... t1^q
        %                                 1 t2 t2^2 ... t2^q
        %                                      ..
        %                                 1 ti ti^2 ... ti^q
        %                                      ..
        %                                 1 tn tn^2 ... tn^q]
        %            q being the number of predictors
        %         Tau : matrix of a hard or fauzzy partition of the data (here for
        %         the RHLP model, Tau is the fuzzy partition represented by the
        %         posterior probabilities (responsibilities) (tik) obtained at the E-Step).
        %
        %         Winit : initial parameter values W(0). dim(Winit) = [(q+1)x(K-1)]
        %         verbose : 1 to print the loglikelihood values during the IRLS
        %         iterations, 0 if not
        %
        % Outputs :
        %
        %          res : structure containing the fields:
        %              W : the estimated parameter vector. matrix of dim [(q+1)x(K-1)]
        %                  (the last vector being the null vector)
        %              piigk : the logistic probabilities (dim [n x K])
        %              loglik : the value of the maximized objective
        %              LL : stored values of the maximized objective during the
        %              IRLS training
        %
        %        Probs(i,gk) = Pro(segment k|cluster g;W)
        %                    = \pi_{ik}(W)
        %                           exp(wgk'vi)
        %                    =  ---------------------------
        %                      1+sum_{l=1}^{K-1} exp(wgl'vi)
        %
        %       with :
        %            * Probs(i,gk) is the prob of component k at time t_i in
        %            cluster g
        %            i=1,...n,j=1...m,  k=1,...,K,
        %            * vi = [1,ti,ti^2,...,ti^q]^T;
        %       The parameter vecrots are in the matrix W=[w1,...,wK] (with wK is the null vector);
        %% References
        % Please cite the following papers for this code:
        %
        %
        % @INPROCEEDINGS{Chamroukhi-IJCNN-2009,
        %   AUTHOR =       {Chamroukhi, F. and Sam\'e,  A. and Govaert, G. and Aknin, P.},
        %   TITLE =        {A regression model with a hidden logistic process for feature extraction from time series},
        %   BOOKTITLE =    {International Joint Conference on Neural Networks (IJCNN)},
        %   YEAR =         {2009},
        %   month = {June},
        %   pages = {489--496},
        %   Address = {Atlanta, GA},
        %  url = {https://chamroukhi.users.lmno.cnrs.fr/papers/chamroukhi_ijcnn2009.pdf}
        % }
        %
        % @article{chamroukhi_et_al_NN2009,
        % 	Address = {Oxford, UK, UK},
        % 	Author = {Chamroukhi, F. and Sam\'{e}, A. and Govaert, G. and Aknin, P.},
        % 	Date-Added = {2014-10-22 20:08:41 +0000},
        % 	Date-Modified = {2014-10-22 20:08:41 +0000},
        % 	Journal = {Neural Networks},
        % 	Number = {5-6},
        % 	Pages = {593--602},
        % 	Publisher = {Elsevier Science Ltd.},
        % 	Title = {Time series modeling by a regression approach based on a latent process},
        % 	Volume = {22},
        % 	Year = {2009},
        % 	url  = {https://chamroukhi.users.lmno.cnrs.fr/papers/Chamroukhi_Neural_Networks_2009.pdf}
        % 	}
        % @article{Chamroukhi-FDA-2018,
        % 	Journal = {},
        % 	Author = {Faicel Chamroukhi and Hien D. Nguyen},
        % 	Volume = {},
        % 	Title = {Model-Based Clustering and Classification of Functional Data},
        % 	Year = {2018},
        % 	eprint ={arXiv:1803.00276v2},
        % 	url =  {https://chamroukhi.users.lmno.cnrs.fr/papers/MBCC-FDA.pdf}
        % 	}
        %
    """
    n,K = Tau.shape
    n,q = M.shape; #q here is (q+1)
    if Winit is None:
        Winit = np.zeros((q, K-1))
    
    I = np.eye(q*(K-1));
    
    #Initialisation du IRLS (iter = 0)
    W_old = Winit;
    
    piik_old, loglik_old = modele_logit(W_old,M,Tau,Gamma);
    
    lmda = 1e-9
    loglik_old = loglik_old - sum(sum(W_old**2))#pow(lmda*(np.linalg.norm(W_old.T.ravel(),2)),2)
    
    iteration = 0;
    converge = False;
    max_iter = 300;
    LL = [];
    
    #utl.globalTrace('IRLS : Iteration {0} Log-vraisemblance {1} \n'.format(iteration, loglik_old))
    
    while not converge and  (iteration<max_iter):
        # Hw_old matrice carree de dimensions hx x hx
        hx = q*(K-1)
        Hw_old = np.zeros((hx,hx))
        gw_old = np.zeros((q, K-1))# todo: verify with matlab this line?
        
        # Gradient :
        for k in range(0, K-1):
            if Gamma is None:
                gwk = np.array([(Tau[:,k] - piik_old[:,k])]).T
            else:
                gwk = Gamma*np.array([(Tau[:,k] - piik_old[:,k])]).T
            for qq in range(0,q):
                vq = M[:,qq]
                gw_old[qq,k] = gwk.T@vq
                
        gw_old = np.array([np.reshape(gw_old,q*(K-1),1)]).T;
        
        
        
        # Hessienne
        for k in range(0, K-1):
            for ell in range(0, K-1):
                delta_kl=int(k==ell) # kronecker delta 
                if Gamma is None:
                    gwk = np.array([piik_old[:,k]]).T*(np.ones((n,1))*delta_kl - np.array([piik_old[:,ell]]).T)
                else:
                    gwk = Gamma*(np.array([piik_old[:,k]]).T*(np.ones((n,1))*delta_kl - np.array([piik_old[:,ell]]).T))
                Hkl = np.zeros((q,q))
                for qqa in range(0,q):
                    vqa=np.array([M[:,qqa]]).T
                    for qqb in range(0,q):
                        vqb=np.array([M[:,qqb]]).T  
                        hwk = vqb.T@(gwk*vqa)
                        Hkl[qqa,qqb] = hwk[0,0]
                        
                
                Hw_old[k*q : (k+1)*q, ell*q : (ell+1)*q] = -Hkl
                
                
        
        # si a priori gaussien sur W (lambda ~ 1e-9)
        Hw_old = Hw_old + lmda*I;
        gw_old = gw_old - np.array([lmda*W_old.T.ravel()]).T
        # Newton Raphson : W(c+1) = W(c) - H(W(c))^(-1)g(W(c))  
        w = np.array([W_old.T.ravel()]).T - np.linalg.inv(Hw_old)@gw_old ; #[(q+1)x(K-1),1]
        W = np.reshape(w,(K-1, q)).T #[(q+1)*(K-1)] 
        #wait=input('enter')
        # mise a jour des probas et de la loglik
        piik, loglik = modele_logit(W, M, Tau ,Gamma)
        
        loglik = loglik - lmda*sum(sum((W**2)))#pow(np.linalg.norm(W.T.ravel(),2),2)
        
        
        
    
        """
         check if Qw1(w^(t+1),w^(t))> Qw1(w^(t),w^(t))
         (adaptive stepsize in case of troubles with stepsize 1) Newton Raphson : W(t+1) = W(t) - stepsize*H(W)^(-1)*g(W)
        """
        pas = 1; # initialisation pas d'adaptation de l'algo Newton raphson
        alpha = 2;
        #print(loglik)
        #print(loglik_old)
        it=0;
        while (loglik < loglik_old):
            pas = pas/alpha; # pas d'adaptation de l'algo Newton raphson
            #recalcul du parametre W et de la loglik
            w = np.array([W_old.T.ravel()]).T - pas*np.linalg.inv(Hw_old)@gw_old
            W = np.reshape(w,(K-1, q)).T
            # mise a jour des probas et de la loglik
            it+=1;
            piik, loglik = modele_logit(W, M, Tau ,Gamma)
            #print('end model logit')
            loglik = loglik - lmda*sum(sum((W**2))) #pow(np.linalg.norm(W.T.ravel(),2),2)
        
        converge1 = abs((loglik-loglik_old)/loglik_old) <= 1e-7
        converge2 = abs(loglik-loglik_old) <= 1e-6
        
        converge = converge1 | converge2
        
        piik_old = piik;
        W_old = W;
        iteration += 1;
        LL.append(loglik_old)
        loglik_old = loglik
        
        #utl.globalTrace('IRLS : Iteration {0} Log-vraisemblance {1} \n'.format(iteration, loglik_old))
        
        
    
    #if converge:
    #    utl.globalTrace('IRLS : convergence  OK ; nbre d''iterations : {0}\n'.format(iteration))
    #else:
    #    utl.globalTrace('\nIRLS : pas de convergence (augmenter le nombre d''iterations > {0}) \n'.format(iteration))
        
     
    if lmda!=0: #pour l'injection de l'a priori dans le calcul de la  loglik de l'EM dans le cas du MAP
        reg_irls = - lmda*pow(np.linalg.norm(W.T.ravel(),2),2)
    else:
        reg_irls = 0
    return W, piik, reg_irls, LL, loglik    

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



    
    
