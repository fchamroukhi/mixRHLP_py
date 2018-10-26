function [klas Z] = MAP(post_probas)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [classes Z_MAP] = MAP(post_probas)
%   
% calcule une partition d'un echantillon par la regle du Maximum A Posteriori à partir des
%
% probabilites a posteriori 
%
% Entrees : post_probas , Matrice de dimensions [n x K] des probabibiltes a
% posteriori (matrice de la partition floue)
%
%       n : taille de l'echantillon
%
%       K : nombres de classes
%
%       klas(i) = arg   max (post_probas(i,k)) , for all i=1,...,n
%                     1<=k<=K
%               = arg   max  p(zi=k|xi;theta)
%                     1<=k<=K
%               = arg   max  p(zi=k;theta)p(xi|zi=k;theta)/sum{l=1}^{K}p(zi=l;theta) p(xi|zi=l;theta)
%                     1<=k<=K
%
% Sorties : classes : vecteur collones contenant les classe (1:K)
%
%       Z : Matrice de dimension [nxK] de la partition dure : ses elements sont zik, avec zik=1 si xi
%       appartient à la classe k (au sens du MAP) et zero sinon.
%
%%%%%%%%%%%%%%%%%%%%%%%%% Faicel Chamroukhi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[N K] = size(post_probas);
[tmp,ikmax] = max(post_probas,[],2);
partition_MAP = (ikmax*ones(1,K))==(ones(N,1)*[1:K]);
klas = ones(N,1);
for k=1:K
%    ll=find(partition_MAP(:,k)==1);
%   klas(ll)=k;
  klas(partition_MAP(:,k)==1)=k;
end
Z = partition_MAP;
