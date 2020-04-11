# Survival-Prediction-Titanic

Python implementation of 4 classification and 1 clustering algorithms to predict the survival of Titanic's passengers.
Classification: Random Forest, K-nn, Neural Networks, Support Vector Machines
Clustering: K-means

Kaggle dataset: https://www.kaggle.com/c/titanic/data

Preprocessing: The attributes that are kept from the dataset are: Pclass, Sex, Age, Sibsp, Parch, Fare and Survived which is the target class. The attributes Pclass ad Sex are being vectorized, Age column has missing values which are filled with the mean value of ages, Sibsp, Parch, Fare are being scaled.

Implementation: The models of each algorithm are fitted on the data and are tested with different values for the parameters. The evaluation is based on accuracy for the classification algortihms and on homogeneity for the clustering one. Higher accuracy and higher homogeneity are prefered.

Results: After trying different combinations of parameter values, the conlcusion is that Random Forest and Neural Networks have higher accuracy. Up next is SVM and last is K-nn, with the lowest accuracy values. About K-means, higher number of clusters give higher homogeneity as well.
