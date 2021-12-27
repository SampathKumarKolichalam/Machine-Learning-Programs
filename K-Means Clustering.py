import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())
df["flower"]=iris.target
print(df.head())
df.drop(["sepal length (cm)","sepal width (cm)","flower"],axis="columns",inplace=True)
df.drop(['sepal length (cm)', 'sepal width (cm)', 'flower'],axis='columns',inplace=True)
print(df.head(3))
km = KMeans(n_clusters=3)
yp = km.fit_predict(df)
print(yp)
df['cluster'] = yp
print(df.head(2))
df.cluster.unique()
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='green')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color='yellow')
plt.show()