from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model=LogisticRegression()
model.fit(X_train,y_train)

score_train=model.score(X_train,y_train)
y_pred=model.predict(X_test)
score_test=model.score(X_test,y_test)
print("Scores for training and test models are: ",score_train," and ",score_test, " Respectively")

MSE=metrics.mean_squared_error(y_pred,y_test)
print("The mean squared error between actual labels and predicted labels on test set are given as ",MSE)
