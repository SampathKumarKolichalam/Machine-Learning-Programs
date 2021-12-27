import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

dataset=load_digits()
print(dataset.keys())
print(dataset.data.shape)
print(dataset.data[0].reshape(8,8))

plt.gray()
plt.matshow(dataset.data[0].reshape(8,8))
plt.show()

print(np.unique(dataset.target))

df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
print(df.head())
print(df.describe())

x=df
y=dataset.target
print(x)
print(y)

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
print(x_scaled)

X_train,X_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=30)

model=LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

pca=PCA(0.95)
X_pca=pca.fit_transform(x)
print(X_pca.shape)
print(pca.explained_variance_ratio_)
print(pca.n_components_)

X_train_pca,X_test_pca,y_train,y_test=train_test_split(X_pca,y,test_size=0.2,random_state=30)
model=LogisticRegression(max_iter=1000)
model.fit(X_train_pca,y_train)
print(model.score(X_test_pca,y_test))

pca=PCA(n_components=2)
X_pca=pca.fit_transform(x)
print(X_pca.shape)

print(pca.explained_variance_ratio_)

X_train_pca,X_test_pca,y_train,y_test=train_test_split(X_pca,y,test_size=0.2,random_state=30)
model=LogisticRegression(max_iter=1000)
model.fit(X_train_pca,y_train)
print(model.score(X_test_pca,y_test))
