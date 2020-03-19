# Python codes for functional data clustering and segmentation with the mixture of regressions with hidden logistic processes (MixRHLP) model: 

<< Note that the codes are also provided in R and Matlab >>

python codes written by

**Faicel Chamrouckhi**
&
**Marius Bartcus**

firstname.lastname@unicaen.fr

For using this code we recommend you install anaconda (which contains the necessary packages to run our codes and a set of useful data science python packages). 
The needed packages to run our code are: NumPy, Scikit-Learn, and matplotlib.

When using this code please cite the following papers : The two first ones concern the model and its use in clustering and the last ones concern the model and its use in discrimination.


```
 @article{Chamroukhi-RHLP-2009,
 	Author = {Chamroukhi, F. and Sam\'{e}, A. and Govaert, G. and Aknin, P.},
 	Journal = {Neural Networks},
 	Number = {5-6},
 	Pages = {593--602},
	Publisher = {Elsevier Science Ltd.},
 	Title = {Time series modeling by a regression approach based on a latent process},
 	Volume = {22},
 	Year = {2009}
     }
 @article{Chamroukhi-MixRHLP-2011,
 	Author = {Sam{\'e}, A. and Chamroukhi, F. and Govaert, G{\'e}rard and Aknin, P.},
 	Issue = 4,
 	Journal = {Advances in Data Analysis and Classification},
 	Pages = {301--321},
 	Publisher = {Springer Berlin / Heidelberg},
 	Title = {Model-based clustering and segmentation of time series with changes in regime},
 	Volume = 5,
 	Year = {2011}
     }

 @article{Chamroukhi-RHLP-FLDA,
 	Author = {Chamroukhi, F. and Sam\'{e}, A. and Govaert, G. and Aknin, P.},
 	Journal = {Neurocomputing},
 	Number = {7-9},
 	Pages = {1210--1221},
 	Title = {A hidden process regression model for functional data description. Application to curve discrimination},
 	Volume = {73},
 	Year = {2010}
     }

 @article{Chamroukhi-FMDA-2013,
 	Author = {Chamroukhi, F. and Glotin, H. and Sam{\'e}, A.},
 	Journal = {Neurocomputing},
 	Pages = {153-163},
 	Title = {Model-based functional mixture discriminant analysis with hidden process regression for curve classification},
 	Volume = {112},
 	Year = {2013}
     }  
@article{Chamroukhi-FDA-2018,
 	Journal = {Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery},
 	Author = {Faicel Chamroukhi and Hien D. Nguyen},
 	Note = {DOI: 10.1002/widm.1298.},
 	Volume = {},
 	Title = {Model-Based Clustering and Classification of Functional Data},
 	Year = {2019},
 	Month = {to appear},
 	url =  {https://chamroukhi.com/papers/MBCC-FDA.pdf}
    }
```


### SHORT DESCRIPTION OF EACH PYTHON FILE. For more detailed description, please see the individual files

1) main_MixFRHLP_EM _Main script to run the EM or CEM algorithm_
2) ModelLearner.py _Contains the two functions of the EM and the CEM algorithm._
3) datasets _Contains the object to load (mainly contains the dataset)_                        
4) MixModel.py _The MixModel class containts the data object and the model settings (number of clusters, the number of regimes, the degree of polynomials, the order of the logistic regression)_
4) MixParam.py _Initializes and updates (the M-step) the model parameters (parameters of the logistic process for each of the clusters, polynomial regression coefficients, and the variances for the regmies for each cluster)._
5) MixStats.py _Calculates mainly the posterior memberships (E-step), the loglikelihood, the parition, different information criterias BIC, ICL, etc_
6) ModelOptions.py _contains some options needed to set before learning the model (like the number of runs, threshold, type of initialization, etc)._
8) enums.py _Used to enumerate the variance type (heteroskedastic or homoscedastic)_
9) RegressionDesinger.py _Design matrices for the polynomial regression and the logistic regression_
10) utils.py _Contains mainly the model_logit function that calculates the pobabilities according to the multinomial logistic model, and an efficient Iteratively Reweighted Least-Squares (IRLS) algorithm.


