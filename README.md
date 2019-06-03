# survival_prediction_titanic

Python implementation of 4 classification and 1 clustering algorithms to predict the survival of Titanic's passengers.
Classification: Random Forest, K-nn, Neural Networks, Support Vector Machines
Clustering: K-means

Kaggle dataset: https://www.kaggle.com/c/titanic/data

Preprocessing: The attributes that are kept from the dataset are: Pclass, Sex, Age, Sibsp, Parch, Fare and Survived which is the target class. The attributes Pclass ad Sex are being vectorized, Age column has missing values which are filled with the mean value of ages, Sibsp, Parch, Fare are being scaled.
