function proba = calcul_proba_modele_Logit(W,X)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% entrées
% W : Matrice des vecteurs paramètre w. W est de dimensions px(K-1) (sans
% le vecteur nul)
% X : Mtrice des variables explicative : de dimensions : pxn -p ordre de
% régression, n nbre d'observations)
% Y : matrice des prédicteurs : les tik
%sorties :
% probabilités

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = size(W,2)+1;
[n,q] = size(X);% q ici c'est (q+1)

eWkTXi = zeros(n,K);
WkTXi = zeros(n,K);

wK=zeros(q,1);
W = [W wK];% ajout du veteur nul pour le calcul des probas

WkTXi= X*W;% multiplication matricielle

%normalisation et calcul des piik

maxm = max(WkTXi,[],2);
WkTXi = WkTXi - maxm*ones(1,K);

eWkTXi = exp(WkTXi);

piik = normalize(eWkTXi,2);
proba = piik;





