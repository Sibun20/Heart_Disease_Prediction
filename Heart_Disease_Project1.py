""" Import all libraries requred for this project """
""" i.e all the dependencies """

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

""" Import the data and preprocess it """

data=pd.read_csv("heart.csv")
df=pd.DataFrame(data)
print(df.info())
print(df.shape)
print(df.head())
print(df.isna().sum())
print(df.describe())

""" 0 -->> person donot have heart disease """
""" 1 -->> person have heart disease """

print(df["target"].value_counts())

""" SNS PLOT """
sns.set_style()
sns.countplot(x="target",data=df,palette="RdBu_r")
plt.show()

"""Split the dataset"""

X=df.drop(columns="target",axis=1)
Y=df["target"]

print("The data after dropping the target column : ", X)
print("The target column ", Y)

"""Spliting the data into training data and testing data"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

model=LogisticRegression()

"""training the model """

model.fit(X_train,Y_train)

""" Model Evaluation """

"""Accuracy on training data """

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy on Training data : ",training_data_accuracy )


"""Accuracy on testing data """

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy on Testing data : ",test_data_accuracy )

""" Build a predictive System """

input_data= (58,0,0,	100,	248,	0,	0,	122,	0,	1,	1,	0,	2)
array_data=np.array(input_data)

""" Reshape the data """

""" Means we want to predicy for one instance or data """

input_data_reshaped = array_data.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)


""" Final Prediction based on the dataset """

if(prediction[0]==0):

    print(" -->> THE PERSON IS COMPLETELY HEALTHY <<-- ")
else:
    
    print(" -->> THE PERSON HAS HEART DISEASE. <<-- ")

""" Graph presentation """
corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(14,17))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.tight_layout()
plt.show()



 
