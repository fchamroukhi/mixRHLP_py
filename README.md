# Introduction to source code for Python codes for functional data clustering and segmentation with mixture of regressions with hidden logistic processes (MixRHLP): 

**Faicel Chamrouckhi**

firstname.lastname@unicaen.fr



For using this code you need to install python.
Also we recomand to install anaconda from: [https://www.anaconda.com/download/] that contains a set of usefull data science python packages.
The packages that we use in our code are:
1) NumPy [http://www.numpy.org/]
2) Scikit-Learn [http://scikit-learn.org/stable/]
3) matplotlib [https://matplotlib.org/]

When using the code 
When using this code please cite the following papers : The two first ones concern the model and its use in clusterng and the two last ones concern the model and its use in discrimination.


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
```


### To obtain a detailed information on each of theese source code:

```
import file_name.py as f
help(f)
```

these gives information on the selected source code with information on classes and functions
each function and class has it's comments that will be given by "help(f)"


### THE SHORT DESCRIPTION OF EACH PYTHON FILE
1) main_MixFRHLP_EM _Script to run the EM or CEM algorithm_
2) ModelLearner.py _Contains two functions that are the EM and CEM algorithm._
3) MixModel.py _Contains the MixModel class with extension of the data object and the number of clusters, the number of regimes, the degree of polynomials, the order of the logistic regression_
4) MixParam.py _Contains the parameters of the logistic process matrix of dimension, polynomial regression coefficient vectors, the variances for the K regmies, logistic proportions for G clusters. Note that the class contains methods to initialize parameters and the M step of the EM algorithm._
5) MixStats.py _Contains the parition, the loglikelihood, different information criterias BIC, ICL, etc. This class computes the E step of the EM algorithm._
6) ModelOptions.py _contains the options needed to set before learning the model._
7) datasets	_Contains the object to load the dataset and setting all the properties used throw the EM or CEM algorithm_                        
8) enums.py _The possible enumerations. In this case we have the variance types enumeration that is free or common_
9) RegressionDesinger.py _Design matrices for the polynomial regression and the logistic regression_
10) utils.py _Contains the model_logit function that calculates the pobabilities according to multinomial logistic model, an efficient Iteratively Reweighted Least-Squares (IRLS) algorithm, and a function for normalizing a matrix.


