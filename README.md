# **Student-Percentage-Prediction (ML)**

> ### GRIP - THE SPARK FOUNDATION
DATA SCIENCE AND BUISNESS ANALYTICS INTERNSHIP

> ### Linear Regression with Python Scikit Learn - Prediction using Supervised ML

## Introduction

**Task-1:** Predict the percentage of an student based on the no. of study hours.

In this regression task we will predict the percentage of marks that a student is expected to score based on the number of hours the studied. this is a simple linear regression task as it involves just two variables.

#### What will be predicted score if a student studies for 9.25 hrs/day? 

[**DataSets**](http://bit.ly/w-data)

## Project

### Import the required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
%matplotlib inline
```
### Reading The Data from data source

```python
student = pd.read_csv("http://bit.ly/w-data")
```
### Data Imported sucessfully

```python
student.head()
```
![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/student.head().png)


```python
student.tail()
```
![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/student.tail().png)


```python
student.describe()
```
![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/student.describe().png)


```python
student.shape
```
![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/student.shape.png)


```python
student.info()
```
![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/student.info().png)

### Checking the missing values

```python
student.isnull().sum()
```
![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/student.isnull().sum().png)


### Checking the correlation between hours and study.

```python
student.corr()
```
![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/student.corr().png)

### Data Visualization
### Visualization with line plot

```python
plt.style.use('ggplot')
student.plot(kind='line')
plt.title('Hours vs Percentage')
plt.xlabel('Hourse studied')
plt.ylabel('Percentage Score')
plt.show()
```
![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/line%20plot.png)

### Data Visualisation with area plot

```python
xmin = min(student. Hours)
xmax = max(student. Hours)
student.plot(kind='area',alpha=0.8, stacked=True, figsize=(15,10),xlim=(xmin, xmax))
plt.title('Hours vs Score',size=15)
plt.xlabel('Hours', size=15)
plt.ylabel('Score',size=15)
plt.show()
```

![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/area%20plot.png)


### Data visualizing with scatter plot

```python
student.plot(kind='scatter',x='Hours', y='Scores', color='g',figsize=(10,8))
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage scores')
plt.show()
```

![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/scatter%20plot.png)


### By Visulization we come to know that this problem can be easily solved by linear regression

### Modeling the data

```python
x = np.asanyarray(student [['Hours']])
y = np.asanyarray(student [[ 'Scores']])

# Using train test split to split the data in train and test Data
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.2, random_state=2)
regressor = LinearRegression()
regressor.fit(train_x, train_y)
print("Training Complete\n")
print('coehhicient: ', regressor.coef_)
print('Intercept: ',regressor.intercept_)
```
![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/Modeling%20the%20data.png)

### we can also plot the fit line over the data in single linear regression

```python
student.plot(kind='scatter',x='Hours',y='Scores', figsize=(5,4),color='r')
plt.plot(train_x, regressor.coef_[0] *train_x + regressor.intercept_,color='b')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()
```

![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/single%20linear%20regression.png)

### The blue line is the best fit line for this data
### Evaluation of model

```python
# Using metrics to find mean obsolute error and r2 to see the accuracy
from sklearn import metrics
from sklearn.metrics import r2_score
y_pred=regressor.predict(test_x)
print('Mean Absolute Error: {}'.format (metrics.mean_absolute_error(y_pred, test_y)))
print("R2-score: %.2f" % r2_score(y_pred, test_y))
```

![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/Evaluation%20of%20model.png)

### Mean absolute Error: it is mean of absolute value of error r2-score: it is not the error but its the metric for accuracy for the model. Higher the r2 value higher is the accuracy of model. Best score is 1

```python
# Comparing Actual vs predicted
# student2 = pd. DataFrame({'Actual': test_y, 'Predicted': y_pred})
# student2
```

### Predicting the score with the single input value

```python
hours = 9.25
predicted_score= regressor.predict([[hours ]])
print (f'No. of hours = {hours}')
print (f'predicted Score = {predicted_score[0]}')
```

![Alt text](https://github.com/Ayush05-pixel/Student-Percentage-Prediction/blob/main/Img/Predicting.png)