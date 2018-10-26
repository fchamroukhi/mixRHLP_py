function misclassif_error_rate =evaluation(true_klas,estimated_klas)
%
%  misclassification error rate in percent
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    crtb=crosstab(true_klas,estimated_klas);
    k=length(crtb);
    a=perms([1:k]);
    for i=1:factorial(k)
        for j=1:k
          Y(i,j)=crtb(j,a(i,j));
        end
    end   
    tab = sum(Y,2);
    maxi = max(tab);
    
n=length(true_klas);
misclassif_examples = n-maxi;
misclassif_error_rate = (misclassif_examples/n)*100;% err en %