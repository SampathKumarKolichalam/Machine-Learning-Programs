import pandas as pd
#from pandas.core.frame import DataFrame
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris=load_iris()
#print(dir(iris))
#print(iris.feature_names)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
#print(df.head())
df["target"]=iris.target
#print(df.head())
##print(iris.target_names)
df["flower_name"]=df.target.apply(lambda x:iris.target_names[x])
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

#sepal:-
plt.xlabel("Sepal Length(cm)")
plt.ylabel("Sepal Width(cm)")
plt.scatter(df0["sepal length (cm)"],df0["sepal width (cm)"],color="green",marker="+")
plt.scatter(df1["sepal length (cm)"],df1["sepal width (cm)"],color="blue",marker="*")
#plt.show()
#Petal:-
plt.xlabel("Petal Length(cm)")
plt.ylabel("Petal Width(cm)")
plt.scatter(df0["petal length (cm)"],df0["petal width (cm)"],color="green",marker="+")
plt.scatter(df1["petal length (cm)"],df1["petal width (cm)"],color="blue",marker="*")
plt.show()

x=df.drop(["target","flower_name"],axis="columns")
#print(x.head())
y=df.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=SVC(gamma=10)
model.fit(x_train,y_train)
#print(model.score(x_test,y_test))
#print(model.predict([[4.8,3.0,1.5,0.3]]))