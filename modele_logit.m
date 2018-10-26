function [probas, loglik] = modele_logit(W,M,Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [probas, loglik] = modele_logit(W,X,Y)
%
% calcule les pobabilites selon un le modele logistique suivant :
%
% probas(i,k) = p(zi=k;W)= \pi_{ik}(W) 
%                        =          exp(wk'vi)
%                          ----------------------------
%                         1 + sum_{l=1}^{K-1} exp(wl'vi)
% for all i=1,...,n et k=1...K (dans un contexte temporel on parle d'un
% processus logistique)
%
% Entrees :
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
%        3. Y Matrice de la partition floue (les probas a posteriori tik)
%           tik = p(zi=k|xi;theta^m); Y de dimensions [nxK] avec K le nombre de classes
% Sorties : 
%
%        1. probas : Matrice de dim [nxK] des probabilites p(zi=k;W) de la vaiable zi
%          (i=1,...,n)
%        2. loglik : logvraisemblance du parametre W du modele logistique
%           loglik = Q1(W) = Esperance(l(W;Z)|X;theta^m) = E(p(Z;W)|X;theta^m) 
%                  = logsum_{i=1}^{n} sum_{k=1}^{K} tik log p(zi=k;W)
%   
% Cette fonction peut egalement �tre utilis�e pour calculer seulement les 
% probas de la fa�oc suivante : probas = modele_logit(W,M)
%
% Faicel Chamroukhi 31 Octobre 2008 (mise � jour)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin > 2
    [n1,K] = size(Y);
    [n2,q] = size(M);% ici q c'est q+1
    if n1==n2
       n=n1;
    else
    error (' W et Y doivent avoir le meme nombre de ligne');
    end
else
     [n,q]=size(M);
end

if nargin > 2
    if size(W,2)== (K-1) % pas de vecteur nul dans W donc l'ajouter
       wK=zeros(q,1);
       W = [W wK];% ajout du veteur nul pour le calcul des probas
    elseif size(W,2)~=K
    error('W doit etre de dimension [(q+1)x(K-1)] ou [(q+1)xK]');
    end
else
    wK=zeros(q,1);
    W = [W wK];% ajout du veteur nul pour le calcul des probas
    [q,K]= size(W);
end

MW = M*W;% multiplication matricielle
maxm = max(MW,[],2);
MW = MW - maxm*ones(1,K);%normalisation
expMW = exp(MW);
piik = expMW./(sum(expMW(:,1:K),2)*ones(1,K));
% piik = normalize(expMW,2);
if nargin>2    %calcul de la log-vraisemblance
    loglik = sum(sum((Y.*MW) - (Y.*log(sum(expMW,2)*ones(1,K))),2));
%     % sans les hig
%     loglik = sum(hig*loglik);
     if isnan(loglik)
        % reglage de pblm numerique exp(MW=-746)=0 et exp(MW=710)=inf)
        MW=M*W;
        minm = -745.1;
        MW = max(MW, minm);
        maxm = 709.78;
        MW= min(MW,maxm);
        expMW = exp(MW);
        
        loglik = sum(sum((Y.*MW) - (Y.*log(sum(expMW,2)*ones(1,K)+eps)),2));        
    end
    if isnan(loglik)
       error('Probleme loglik IRLS NaN (!!!)');
    end
else
    loglik = [];
end
probas = piik;



