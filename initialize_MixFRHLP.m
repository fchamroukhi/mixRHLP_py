function param = initialize_MixFRHLP(data,G , K, phiBeta, phiW, variance_type, init_kmeans, try_algo) 
%%%%%%%%%%%%%%%%%%%%
%
%essai_EM
%
%
%
%%%%%%%%%%%%%%%%%% FC
[n m]=size(data);
p = size(phiBeta,2)-1;
q = size(phiW,2)-1;

%Initialization of the model parameters for each cluster: W (pi_jgk), betak and sigmak  
% 1. Initialization of cluster weights
param.alpha_g=1/G*ones(G,1);

% 2. betagk and sigmagk
if init_kmeans
    D = data;       
    max_iter_kmeans = 400;
    n_tries_kmeans = 20;
    verbose_kmeans = 0; 
    
    res_kmeans = Kmeans_faicel(D,G,n_tries_kmeans, max_iter_kmeans, verbose_kmeans);
    
    klas = res_kmeans.klas;
    
    for g=1:G
        Xg = D(klas==g ,:); %if kmeans            
        param_init =  init_regression_param(Xg, K,phiBeta, variance_type, try_algo);   
        param.beta_g(:,:,g) = param_init.betak;
        if strcmp(variance_type,'common')
            param.sigma_g(g) = param_init.sigma;
        else
            param.sigma_g(:,g) = param_init.sigmak;
        end
    end
else
    klas = zeros(n,1);
    ind = randperm(n);
    D=data;
    for g=1:G
        if g<G
            ind_klasg=ind((g-1)*round(n/G) +1 : g*round(n/G));
            klas(ind_klasg)=g;
            Xg = D(ind_klasg,:);
        else%g=G
            ind_klasG = ind((g-1)*round(n/G) +1 : end);
            klas(ind_klasG)=G;
            Xg = D(ind_klasG,:);
        end
        
        param_init =  init_regression_param(Xg, K,phiBeta, variance_type, try_algo);
        
        param.beta_g(:,:,g) = param_init.betak;
        
        if strcmp(variance_type,'common')
            param.sigma_g(g) = param_init.sigma;
        else
            param.sigma_g(:,g) = param_init.sigmak;
        end
    end
end

% 3.  W (pi_jgk)  
% for g=1:G
%     clusterg_labels =  repmat(klas,1,m)';% [m x n]
%     clusterg_labels = clusterg_labels(:);
    [W pijgk] = init_hlp(G, K, q, klas, phiW, try_algo);
    param.Wg = W;
    param.pi_jgk = pijgk;
%     param.pi_jgk(:,:,g) = repmat(pijgk(1:m,:,g),n,1);
% end



%%%%%%%%%%%%%%%%%%%%%%%
function [Wg  pi_jgk] =  init_hlp(G, K, q, klas, phiW, essai_EM)

% init_hlp initialize the Hidden Logistic Process
%
%
%
%%%%%%%%%%%% FC %%%%%%%
% %   1. Initialisation de W (pi_jgk)

[nm q1] = size(phiW);
n = length(klas);
m = nm/n;

clusterg_labels =  repmat(klas,1,m)';% [m x n]
clusterg_labels = clusterg_labels(:);
    
pi_jgk = zeros(nm,K,G);

Wg = zeros(q+1,K-1,G);
if (essai_EM) ==1
    for g=1:G
        %Wg(:,:,g) = zeros(q+1,K-1);%initialisation avec le vect null du vercteur param�tre du IRLS
        pijgk = modele_logit(Wg(:,:,g),phiW(clusterg_labels==g,:));
        pi_jgk(:,:,g) = repmat(pijgk(1:m,:),n,1);
    end
else
    for g=1:G
        Wg(:,:,g) = rand(q+1,K-1);%initialisation aléatoire du vercteur param�tre du IRLS
        pijgk = modele_logit(Wg(:,:,g),phiW(clusterg_labels==g,:));
        pi_jgk(:,:,g) = repmat(pijgk(1:m,:),n,1);
    end
end

    
%%%%%%%%%%%%%%%%%%%%%%
function para = init_regression_param(data,K, phi,type_variance, essai_EM)
 
% init_regression_param initialize the Regresssion model with Hidden Logistic Process
%
%
%
%%%%%%%%%%%%%%%%%%%% Faicel Chamroukhi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n m] = size(data);
X = data;
 
if strcmp(type_variance,'common')
    s=0;
end 
%
 
 if (essai_EM) ==1 %2
     
     %decoupage de l'echantillon (signal) en K segments
     zi = round(m/K)-1;
     for k=1:K
         i = (k-1)*zi+1;
         j = k*zi;
         
         Xij = X(:,i:j);
         Xij = reshape(Xij',[],1);
        
        phi_ij = phi(i:j,:);
        Phi_ij=repmat(phi_ij,n,1);
           
        bk = inv(Phi_ij'*Phi_ij)*Phi_ij'*Xij; 
        
        para.betak(:,k) = bk;
        
         if strcmp(type_variance,'common')
            para.sigma = var(Xij);%1000;
         else
             mk = j-i+1 ;%length(Xij);
             z = Xij-Phi_ij*bk;
             sk = z'*z/(n*mk); 
             para.sigmak(k) = sk;
            %para.sigmak(k) = var(Xij);
         end
     end
 else % initialisation aléatoire
     Lmin= round(m/(K+1));%nbr pts min dans un segments
     tk_init = zeros(1,K+1);
     tk_init(1) = 0;         
     K_1=K;
     for k = 2:K
         K_1 = K_1-1;
         temp = tk_init(k-1)+Lmin:m-K_1*Lmin;
         ind = randperm(length(temp));
         tk_init(k)= temp(ind(1));                      
     end
     tk_init(K+1) = m;
     %model.tk_init = tk_init;
     for k=1:K
         i = tk_init(k)+1;
         j = tk_init(k+1);
         Xij = X(:,i:j);
         Xij = reshape(Xij',[],1);
        
        phi_ij = phi(i:j,:);
        Phi_ij=repmat(phi_ij,n,1);
           
        bk = inv(Phi_ij'*Phi_ij)*Phi_ij'*Xij; 
        para.betak(:,k) = bk;
        
         if strcmp(type_variance,'common')
            para.sigma = var(Xij);%1000;
         else
             mk = j-i+1 ;%length(Xij);
             z = Xij-Phi_ij*bk;
             sk = z'*z/(n*mk); 
             para.sigmak(k) = sk;
            %para.sigmak(k) = var(Xij);
         end
     end
     
 end
