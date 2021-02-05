import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

my_data = np.genfromtxt(filepath, delimiter=',',skip_header=True)
labels=my_data[:,0] # First column of dataset contains MNIST labels
pixels=my_data[:,1:]# Remaining columns consist of pixel values
pixels=pixels.reshape((-1,28*28))#Reshaping as Logistic Regression expects 2D data rather than 3D data (-1,28,28) isn't feasible
labels=labels.reshape((-1,1))#Converting (42000,) to (4200,1)

X_train, X_test, y_train, y_test = train_test_split(pixels,labels, test_size=0.33, random_state=42)
