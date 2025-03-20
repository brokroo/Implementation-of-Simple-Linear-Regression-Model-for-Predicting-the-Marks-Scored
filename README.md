# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SANJITH.R
RegisterNumber: 212223230191 
```

```
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df=pd.read_csv('/Users/home/...................../EX02/student_scores.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
x=df.drop(columns='Scores')
y=df[['Scores']]
print(type(x))
print(type(y))
print(y)
print(x)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
X_train.shape
model=LinearRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
plt.scatter(x,y,label='Actual Data',color='black')
plt.plot(x,model.predict(x),color='purple')
plt.show()
mse=mean_squared_error(Y_test,y_pred)
print(f"Mean Squarred Error is :  {mse}")
mae=mean_absolute_error(Y_test,y_pred)
print(f"Mean Abslute Error is :  {mae}")
rmse=np.sqrt(mse)
print(f"Root Mean Squarred Error is :  {rmse}")
a=np.array([[13]])
yp=model.predict(a)
print(yp)           # 124-132 marks estimated
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
