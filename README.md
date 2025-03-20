# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### 1.Import the standard Libraries.
### 2.Set variables for assigning dataset values.
### 3.Import linear regression from sklearn.
### 4.Assign the points for representing in the graph.
### 5.Predict the regression for marks by using the representation of the graph.
### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

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
## DATA
![image](https://github.com/user-attachments/assets/3d905aca-8e59-4762-ba5a-d8497036e169)

## X AND Y:
![image](https://github.com/user-attachments/assets/6f149ecc-d22d-45ce-a471-ed77d6cb6986)

## train_test_split:
![image](https://github.com/user-attachments/assets/43b52350-7746-4598-8e69-4517813f0ef2)

## LinearRegression Model:
![image](https://github.com/user-attachments/assets/f040dd7b-b425-43c0-9e78-e2765789f359)

## Visualization:
![image](https://github.com/user-attachments/assets/6c39aa81-2ecf-4646-809c-d3b2f34b4a8f)

## Error Analysis:
![image](https://github.com/user-attachments/assets/c025642c-aa20-4b36-a4e4-bc44ce1a22db)

## Predictive System:
![image](https://github.com/user-attachments/assets/bfd97498-1120-4ee6-bbe8-8f5c59549139)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
