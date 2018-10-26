function solution =  MixFRHLP_EM(data, G , K, p, q, ... 
    variance_type, init_kmeans, total_EM_tries, max_iter_EM, threshold, verbose, verbose_IRLS) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% solution =  curve_clustering_MixFRHLP_EM(data,G , K, p, q, fs,[options])
%
% Learn a MixFRHLP model for curve clustering by EM
%
%
% Inputs : 
%
%          1. data :  n curves each curve is composed of m points : dim(X)=[n m] 
%                * Each curve is observed during the interval [0,T]=[t_1,...,t_m]
%                * t{j}-t_{j-1} = 1/fs (fs: sampling period)    
%          2. G: number of clusters
%          3. K: Number of polynomial regression components (regimes)
%          4. p: degree of the polynomials
%
% Options:
%          1. q:  order of the logistic regression (by default 1 for convex segmentation)
%          2. variance_type of the poynomial models for each cluster (free or
%          common, by defalut free)
%          3. init_kmeans: initialize the curve partition by Kmeans
%          4. total_EM_tries :  (the solution providing the highest log-lik is chosen
%          5. max_iter_EM
%          6. threshold: by defalut 1e-6
%          7. verbose : set to 1 for printing the "complete-log-lik"  values during
%          the EM iterations (by default verbose_EM = 0)
%          8. verbose_IRLS : set to 1 for printing the values of the criterion 
%             optimized by IRLS at each IRLS iteration. (IRLS is used at
%             each M step of the EM algorithm). (By defalut: verbose_IRLS = 0)
%
% Outputs : 
%
%          solution : structure containing the following fields:
%                   
%          1. param : a structure containing the model parameters
%                       ({Wg},{alpha_g}, {beta_gk},{sigma_gk}) for g=1,...,G and k=1...K. 
%              1.1 Wg = (Wg1,...,w_gK-1) parameters of the logistic process:
%                  matrix of dimension [(q+1)x(K-1)] with q the order of logistic regression.
%              1.2 beta_g = (beta_g1,...,beta_gK) polynomial regression coefficient vectors: matrix of
%                  dimension [(p+1)xK] p being the polynomial  degree.
%              1.3 sigma_g = (sigma_g1,...,sigma_gK) : the variances for the K regmies. vector of dimension [Kx1]
%              1.4 pi_jgk :logistic proportions for cluster g
%
%          2. Psi: parameter vector of the model: Psi=({Wg},{alpha_g},{beta_gk},{sigma_gk}) 
%                  column vector of dim [nu x 1] with nu = nbr of free parametres
%          3. h_ig = prob(curve|cluster_g) : post prob (fuzzy segmentation matrix of dim [nxG])
%          4. c_ig : Hard partition obtained by the AP rule :  c_{ig} = 1
%                    if and only c_i = arg max_g h_ig (g=1,...,G)
%          5. klas : column vector of cluster labels
%          6. tau_ijgk prob(y_{ij}|kth_segment,cluster_g), fuzzy
%          segmentation for the cluster g. matrix of dimension
%          [nmxK] for each g  (g=1,...,G).
%          7. Ex_g: curve expectation: sum of the polynomial components beta_gk ri weighted by 
%             the logitic probabilities pij_gk: Ex_g(j) = sum_{k=1}^K pi_jgk beta_gk rj, j=1,...,m. Ex_g 
%              is a column vector of dimension m for each g.
%          8. comp-loglik : at convergence of the EM algo
%          9. stored_com-loglik : vector of stored valued of the
%          comp-log-lik at each EM teration 
%          
%          10. BIC value = loglik - nu*log(nm)/2.
%          11. ICL value = comp-loglik_star - nu*log(nm)/2.
%          12. AIC value = loglik - nu.
%          13.   log_alphag_fg_xij 
%          14.   polynomials 
%          15.   weighted_polynomials 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Faicel Chamroukhi (septembre 2009) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 5, q=1; end
if nargin < 6, q=1; variance_type = 'free';end
if nargin < 7, q=1; variance_type = 'free';total_EM_tries = 10;end
if nargin < 8, q=1; variance_type = 'free';total_EM_tries = 10; init_kmeans = 1;end
if nargin < 9, q=1; variance_type = 'free';total_EM_tries = 10; init_kmeans = 1; max_iter_EM = 1000;end
if nargin < 10, q=1; variance_type = 'free';total_EM_tries = 10; init_kmeans = 1; max_iter_EM = 1000; threshold = 1e-6; end
if nargin < 11, q=1; variance_type = 'free';total_EM_tries = 10; init_kmeans = 1; max_iter_EM = 1000; threshold = 1e-6; verbose = 0; end
if nargin < 12, q=1; variance_type = 'free';total_EM_tries = 10; init_kmeans = 1; max_iter_EM = 1000; threshold = 1e-6; verbose = 0; verbose_IRLS = 0; end
warning off

% if strcmp(variance_type,'common')
%     common_variance =1;
% else
%     common_variance=0;
% end
        
[n, m] = size(data);%n  nbre de signaux (individus); m: nbre de points pour chaque signal
% % construction des matrices de regression
x = linspace(0,1,m);% ou rentrer le vecteur de covariables des courbes
% pour 1 courbe
[phiBeta, phiW] = designmatrix_FRHLP(x, p, q);
%pour les n courbes (regularly sampled)
phiBeta = repmat(phiBeta, n, 1);
phiW = repmat(phiW, n, 1);
%
X=reshape(data',[],1);

top=0;
%% % main algorithm
try_EM = 0;
best_loglik = -inf;
cputime_total = [];
while try_EM < total_EM_tries
    try_EM = try_EM +1;
    fprintf('EM try n� %d\n',try_EM);
    time = cputime;    
    
    % % Initialisation 
    
    param = initialize_MixFRHLP_EM(data, G , K, phiBeta, phiW, variance_type, init_kmeans, try_EM);
    %
    
    iter = 0; 
    converge = 0;
    prev_loglik = -inf;
     %% EM 
    tau_ijgk = zeros(n*m,K,G);%% segments post prob  
    log_tau_ijgk = zeros(n*m,K,G);
    %
    log_fg_xij = zeros(n,G); 
    log_alphag_fg_xij = zeros(n,G); 
    while ~converge &&(iter<= max_iter_EM)
            %%%%%%%%%%%
            % E-Step  %
            %%%%%%%%%%%
%         log_sum_pijgk_fgk_xij = zeros(n*m,G);
%         log_prod_sum_pijgk_fgk_xij = zeros(n,G);

        for g=1:G
            alpha_g = param.alpha_g; % cluster weights
            beta_g = param.beta_g(:,:,g);
            Wg = param.Wg(:,:,g);
            pi_jgk = param.pi_jgk(:,:,g);
            %% calcul de log_tau_ijgk (par log_pijgk_fgk_xij pour xi)
            log_pijgk_fgk_xij =zeros(n*m,K);
            for k = 1:K
                beta_gk = beta_g(:,k);
                if strcmp(variance_type,'common')
                    sgk = param.sigma_g(g);
                else
                    sgk = param.sigma_g(k,g);
                end                       
                z=((X-phiBeta*beta_gk).^2)/sgk;
                log_pijgk_fgk_xij(:,k) = log(pi_jgk(:,k)) - 0.5*(log(2*pi)+log(sgk)) - 0.5*z;%pdf cond à c_i = g et z_i = k de xij
                %pijgk_fgk_xij(:,k) = pi_jgk(:,k).*normpdf(X,phiBeta*beta_gk,sqrt(sigma_g(k)));%---%pdf cond à c_i = g et z_i = k de xij
            end   
 
            log_pijgk_fgk_xij = min(log_pijgk_fgk_xij,log(realmax));
            log_pijgk_fgk_xij = max(log_pijgk_fgk_xij,log(realmin));
                      
            pijgk_fgk_xij = exp(log_pijgk_fgk_xij);
            sumk_pijgk_fgk_xij = sum(pijgk_fgk_xij,2); % sum over k
            log_sumk_pijgk_fgk_xij  = log(sumk_pijgk_fgk_xij); %[nxm x 1]
            
            log_tau_ijgk(:,:,g) = log_pijgk_fgk_xij - log_sumk_pijgk_fgk_xij *ones(1,K);
            tau_ijgk(:,:,g) = exp(log_normalize(log_tau_ijgk(:,:,g))); 
            % likelihood for each curve
            % fg_xij = prod(sumk_pijgk_fgk_xij(:,:,g) ,2);% [n x 1]:  prod over j=1,...,m: fg_xij = prod_j sum_k pi_{jgk} N(x_{ij},mu_{gk},s_{gk))
            
            % log-lik for the n_g curves of cluster g
            log_fg_xij(:,g) = sum(reshape(log_sumk_pijgk_fgk_xij,m,n)',2);% [n x 1]:  sum over j=1,...,m: fg_xij = prod_j sum_k pi_{jgk} N(x_{ij},mu_{gk},s_{gk))
            log_alphag_fg_xij(:,g) = log(alpha_g(g)) + log_fg_xij(:,g);% [nxg] 
        end      
        log_alphag_fg_xij = min(log_alphag_fg_xij,log(realmax));
        log_alphag_fg_xij = max(log_alphag_fg_xij,log(realmin));

        % cluster posterior probabilities p(c_i=g|X)
        %h_ig = alphag_fg_xij./(sum(alphag_fg_xij,2)*ones(1,G));% [nxg]
        h_ig = exp(log_normalize(log_alphag_fg_xij)); 
        
        % log-likelihood
        loglik = sum(log(sum(exp(log_alphag_fg_xij),2)));% + res.reg_irls;
        %%%%%%%%%%%
        % M-Step  %
        %%%%%%%%%%%          
         
        % Maximization w.r.t alpha_g
        param.alpha_g = sum(h_ig,1)'/n;
    
        % Maximization w.r.t betagk et sigmagk
%         cluster_labels =  repmat(klas,1,m)';% [m x n]
%         cluster_labels = cluster_labels(:);         
        for g=1:G 
            temp =  repmat((h_ig(:,g)),1,m);% [m x n]
            cluster_weights = reshape(temp',[],1);%cluster_weights(:)% [mn x 1]    
            tauijk = tau_ijgk(:,:,g);%[(nxm) x K]
            if strcmp(variance_type,'common'),  s = 0; 
            else
                    sigma_gk = zeros(K,1);
            end
            
            for k=1:K    
                segment_weights = tauijk(:,k);%poids du kieme segment   pour le cluster g                  
                % poids pour avoir K segments floues du gieme cluster flou 
                phigk = (sqrt(cluster_weights.*segment_weights)*ones(1,p+1)).*phiBeta;%[(n*m)*(p+1)]
                Xgk = sqrt(cluster_weights.*segment_weights).*X;  
                %% maximization w.r.t beta_gk: Weighted least squares 
                beta_gk(:,k) = inv(phigk'*phigk + eps*eye(p+1))*phigk'*Xgk; % Maximization w.r.t betagk
                %    the same as
                %                 W_gk = diag(cluster_weights.*segment_weights);
                %                 beta_gk(:,k) = inv(phiBeta'*W_gk*phiBeta)*phiBeta'*W_gk*X;
                %                
                %% Maximization w.r.t au sigma_gk :                  
                if strcmp(variance_type,'common')
                    sk = sum((Xgk-phigk*beta_gk(:,k)).^2);
                    s = s+sk;
                    %ng*m = sum(sum(tauijk)) with ng = length(find(cluster_labels==g))/m;%ng = size(reshape(Xg,[],m))%
                    sigma_gk = s/sum(sum((cluster_weights*ones(1,K)).*tauijk));
                else
                    sigma_gk(k)= sum((Xgk-phigk*beta_gk(:,k)).^2)/(sum(cluster_weights.*segment_weights));
                end
            end
            param.beta_g(:,:,g) = beta_gk;
            param.sigma_g(:,g) = sigma_gk; 

            % Maximization w.r.t W 
            %%  IRLS : Regression logistique multinomiale pondérée par
            %%  cluster
            Wg_init = param.Wg(:,:,g);
            
            %clus_seg_weights=(cluster_weights*ones(1,K)).*tauijk;
            %clus_seg_weights=normalize(clus_seg_weights,2);%%!!!!!
            %res = IRLS(clus_seg_weights, phiW, Wg_init, verbose_IRLS);
            
            res = IRLS_MixFRHLP(cluster_weights,tauijk, phiW, Wg_init, verbose_IRLS);
                       
            param.Wg(:,:,g)=res.wk;             
            param.pi_jgk(:,:,g) = repmat(res.piik(1:m,:),n,1); 
        end
        %% Fin EM
        
        iter=iter+1;        
        
        if verbose, 
            fprintf(1,'EM   : Iteration : %d   log-likelihood : %f \n',  iter, loglik); 
        end
        if prev_loglik - loglik > 1e-5
            fprintf(1, '!!!!! EM log-likelihood is decreasing from %6.4f to %6.4f!\n', prev_loglik, loglik);
            top = top+1;   
            if top>20 ,  break; %solution =  curve_clustering_MixFRHLP_EM(data,G , K, p,fs, q, ... variance_type, init_kmeans, total_EM_tries, max_iter_EM, threshold, verbose, verbose_IRLS) ;
            end
        end
        %test of convergence
        converge = abs((loglik-prev_loglik)/prev_loglik) <= threshold;
        prev_loglik = loglik;
        stored_loglik(iter) = loglik;         
    end% fin  EM  loop   
    cputime_total = [cputime_total cputime-time];
    
    solution.param = param; 
    solution.Psi = [param.alpha_g(:); param.Wg(:); param.beta_g(:); param.sigma_g(:)];
 
    solution.param.pi_jgk = param.pi_jgk(1:m,:,:);
    solution.h_ig = h_ig;
    solution.tau_ijgk = tau_ijgk;
    solution.loglik = loglik;
    solution.stored_loglik = stored_loglik;
    solution.log_alphag_fg_xij = log_alphag_fg_xij;
    
    if loglik > best_loglik
       best_solution = solution;
       best_loglik = loglik;
       %solution.loglik = loglik;
    end         
    if total_EM_tries>1,  fprintf(1,'max value: %f \n',solution.loglik);    end
    
end 
solution = best_solution;

[solution.klas c_ig] = MAP(solution.h_ig); % c_ig the hard partition of the curves 
%
if total_EM_tries>1,  fprintf(1,'max value: %f \n',solution.loglik);    end
%
for g=1:G 
    solution.polynomials(:,:,g) = phiBeta(1:m,:)*solution.param.beta_g(:,:,g);
    solution.weighted_polynomials(:,:,g) = solution.param.pi_jgk(:,:,g).*solution.polynomials(:,:,g);
    solution.Ex_g(:,g) = sum(solution.weighted_polynomials(:,:,g),2); 
end
%
solution.Ex_g = solution.Ex_g(1:m,:); 
solution.cputime = mean(cputime_total);
% loglik = sum(sum(solution.log_alphag_fg_xij,2));
% solution.phiBeta = phiBeta(1:m,:);
% solution.phiW = phiW(1:m,:);

nu = length(solution.Psi);
% BIC AIC et ICL*
solution.BIC = solution.loglik - (nu*log(n)/2);%n*m/2!
solution.AIC = solution.loglik - nu;
% ICL*             
% Compute the comp-log-lik 
cig_log_alphag_fg_xij = (c_ig).*(solution.log_alphag_fg_xij);
comp_loglik = sum(sum(cig_log_alphag_fg_xij,2)); 

solution.ICL1 = comp_loglik - nu*log(n)/2;%n*m/2!

