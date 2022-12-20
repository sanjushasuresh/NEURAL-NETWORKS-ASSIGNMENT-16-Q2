# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 21:18:39 2022

@author: LENOVO
"""

# pip install keras 
# pip install tensorflow

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd

df=pd.read_csv("forestfires.csv",delimiter=',')
df.head()

# Splitting
X=df.iloc[:,:30]
X.columns
X.dtypes

Y=df["size_category"]


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
X["month"]=LE.fit_transform(X["month"])
X["month"]=pd.DataFrame(X["month"])

X["day"]=LE.fit_transform(X["day"])
X["day"]=pd.DataFrame(X["day"])

df["size_category"]=LE.fit_transform(df["size_category"])
Y=pd.DataFrame(df["size_category"])
Y

model=Sequential()
model.add(Dense(45,input_dim=30,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

history=model.fit(X,Y,validation_split=0.33,epochs=250,batch_size=10)
scores=model.evaluate(X,Y)
print("%s: %.2f%%"%(model.metrics_names[1],scores[1]*100))
history.history.keys()
# accuracy=98.65%


import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.show()


#=============================================================================================#


# Method 2

import numpy as np
import pandas as pd

df=pd.read_csv("forestfires.csv",delimiter=',')
df.head()
df.isnull().sum()
df.shape
df.duplicated()
df[df.duplicated()]
df.drop_duplicates(inplace=True)
df.columns
df.dtypes


# Splitting
X=df.iloc[:,:30]
X.columns
X.dtypes

Y=df["size_category"]

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
X["month"]=LE.fit_transform(X["month"])
X["month"]=pd.DataFrame(X["month"])

X["day"]=LE.fit_transform(X["day"])
X["day"]=pd.DataFrame(X["day"])

df["size_category"]=LE.fit_transform(df["size_category"])
Y=pd.DataFrame(df["size_category"])
Y


# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30))
mlp.fit(X_train,Y_train)
pred_train=mlp.predict(X_train)
pred_test=mlp.predict(X_test)

from sklearn.metrics import accuracy_score
as1=accuracy_score(Y_train,pred_train)
# as1=0.98033 (98%)
as2=accuracy_score(Y_test,pred_test)
# as2=0.95711 (96%)