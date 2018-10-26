function res = Kmeans_faicel(X,K,nbre_essais,nbre_iter_max,verbose)
%    
%
%   Algorithme des K-means
%   
%   X de dim nxp 
%   
%
% Faicel CHAMROUKHI Septembre 2008 (mise a jour)
%
%
%
% distance euclidienne
%
%%%%%%%%%%%%%%%%%%%%%%%%%% Faice Chamroukhi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Algo des kmeans
if nargin<5
    verbose=0;
end
if nargin <4
   nbre_iter_max = 100;
end
if nargin<3
   nbre_essais = 20;
end


[n p] = size(X);
%% cas d'une seule classe
moy_globale = mean(X,1);

if K==1
   dmin = sum((X-ones(n, 1)*moy_globale).^2,2);
   res.pik=1;
   res.cg = moy_globale;
   res.sigmag = ((X - ones(n,1)*moy_globale)'*(X- ones(n,1)*moy_globale))/n;
   klas = ones(n,1);
   res.klas = klas;
   res.err = sum(dmin);
   res.Zik = ones(n,1);
   return;
end;  


nbr_essai = 0;
best_solution.err= inf; 
nb_good_try=0; 
total_nb_try=0;
while (nbr_essai<nbre_essais)
     nbr_essai=nbr_essai + 1;
     if verbose,  fprintf('Kmeans : Essai %d  \n',nbr_essai); end
     critere =[];%distortion
     iter=0;
     partition = zeros(n,K);%initialisation de la partition
     Zik = zeros(n,K);
     partition_inchangee = 0;
     err=-inf;
     %% 1.Initialisation aleatoire des centres
     indices_aleatoires = randperm(n);
     centres = X(indices_aleatoires(1:K),:);% 
     while (iter<=nbre_iter_max && ~partition_inchangee)
         iter = iter+1;
         clas_vide=[];
         old_centres = centres;
         
         % calcul des distances euclidiennes
         dist_eucld = zeros(n,K);%distanre entre chaque xi et mk
         for k = 1:K
             mk = centres(k,:);
             dist_eucld(:,k) = sum((X-ones(n,1)*mk).^2,2);
         end
%% Etape affectation (recalcul de la partition)

            [dmin klas] = min(dist_eucld');

            klas = klas';
            Zik = (klas*ones(1,K))==(ones(n,1)*[1:K]);
%% Etape representation (recalcul des centres)
        sigmak = zeros(p,p,K);
        pik = zeros(K,1);
        for m=1:K            
            ind_ck = find(klas ==m);
            
            %cas de classe vide
            if isempty(ind_ck),
                clas_vide =[clas_vide;m];
            else

            classek = X(ind_ck,:);
            %estimation des proportions
            nk = length(ind_ck);
            pik(m) = nk/n;
            %estimation des centres
            centres(m,:) = mean(classek,1);
            
            %estimation des variance
            mk = centres(m,:);
            sigmak(:,:,m)= ((classek - ones(nk,1)*mk)'*(classek- ones(nk,1)*mk))/nk;
            end
        end
%         centres(clas_vide,:)=[];
        centres(clas_vide,:)= old_centres(clas_vide,:);
        
        %% 1. premier critere d'arret : distortion inferieure a un seuil
        %% petit
        err2 = sum(sum(Zik.*dist_eucld,2));% err2 = sum(dmin); %distortion
        crit1 = (abs(err2-err))/err <1e-6;
        %% 2. deuxieme critere : les centres ne varient pas

        crit2 = max(max(abs(centres - old_centres)))<1e-6;
       
        if crit1 | crit2
            partition_inchangee = 1; % convergence
        end
        
        err = err2;
        if verbose
            fprintf('Kmeans : Iteration  %d  Critere %6f  \n',iter,err);
        end

     end% boucle du kmeans
%%  la solution a chaque essai    
res.pik = pik;
res.cg = centres;
res.sigmag = sigmak; %variances
res.Zik = Zik;
res.klas = klas;
res.err = err;
%%  Pour avoir une meilleure solution, faire  plusieurs essais
    %if ~isempty(mk)
    %   nb_good_try = nb_good_try+1; 
     %  total_nb_try = 0;
       if err < best_solution.err
          best_solution = res;
       end
    %end

    %if total_nb_try > 500
    %  fprintf('Impossible d''obtenir ce nombre de classes \n');
    %  res=[];
    %  return   % ou continue
    %end
end %boucle sur les essais

res = best_solution;
% fprintf(' val opt du critere %6f  \n',res.err);

