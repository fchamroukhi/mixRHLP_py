function [phiBeta, phiW] = designmatrix_FRHLP(t, p, q)
%
%
%
%
%
%
%
%
%
%
%
%
%
%
% Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(t,1) == 1;
    t=t'; % en vecteur
end

if nargin > 2
    ordre_max = max(p,q);
else
    ordre_max = p;
end

phi=[];
for ord = 0:ordre_max
    phi =[phi, t.^(ord)];% phi2w = [1 t t.^2 t.^3 t.^p;......;...]
end
phiBeta= phi(:,1:p+1); %matrice de regresseurs pour Beta

if nargin > 2
   phiW = phi(:,1:q+1);% matrice de regresseurs pour w
else
    phiW =[];
end