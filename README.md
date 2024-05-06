# Implementation-of-SVM-For-Spam-Mail-Detection

# AIM:
To write a program to implement the SVM For Spam Mail Detection.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

# Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: KANIMOZHI
RegisterNumber:  212222230060
*/

import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy  
```

# Output:
## RESULT
![6](https://github.com/kanimozhipannerselvam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476060/d98e28eb-9209-40dd-aeca-bb672098bc05)

## data.head()
![1](https://github.com/kanimozhipannerselvam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476060/30271d24-9832-4583-9eff-f8277b0e4c6b)

## data.info()
![2](https://github.com/kanimozhipannerselvam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476060/2e72d0cb-20eb-4f2b-a00d-1848fde27113)

## data.isnull().sum()
![3](https://github.com/kanimozhipannerselvam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476060/cb7722c9-af45-42b3-9f17-50f9f9686187)

## predicted values
![4](https://github.com/kanimozhipannerselvam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476060/8241b87c-dfc1-4c99-a803-061ac13bb416)

## Accuracy
![5](https://github.com/kanimozhipannerselvam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476060/e8a0112c-6c34-420a-b67f-814c7071f1c5)


# Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
