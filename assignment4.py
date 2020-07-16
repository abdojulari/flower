# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset =  pd.read_csv('datasets_19_420_Iris.csv')

# supervised learning algorithm 

X  = dataset[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']]
Y  = dataset.Species
 
# Splitting the dataset into 2 in proportion of 3:7  

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size=.3)

model = DecisionTreeClassifier() # building the model
fittedModel = model.fit(X_Train, Y_Train) # fitting the model with training sets for classification

# predictions for classification 
predictions = fittedModel.predict(X_Test)

print(confusion_matrix(Y_Test, predictions))

# accuracy score to check the efficiency of the model 

print(accuracy_score(Y_Test, predictions)) # ascertain the percentage of the that is classfied correctly  