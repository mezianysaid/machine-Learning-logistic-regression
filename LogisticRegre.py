from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

df=load_iris()
X=df.data
Y=df.target
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
model=linear_model.LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(df.target_names[y_pred])
print(accuracy_score(y_test,y_pred))
