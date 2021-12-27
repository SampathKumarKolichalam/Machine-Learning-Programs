import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())
df['target'] = iris.target
print(df.head())
print(df[df.target==1].head())
print(df[df.target==2].head())
df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
print(df.head())

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')
plt.show()

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')
plt.show()

#Train-Test_split:-
X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(len(X_train))
print(len(X_test))

#KNN Classifier:-
knn = KNeighborsClassifier(n_neighbors=10)
print(knn.fit(X_train, y_train))
print(knn.score(X_test, y_test))
print(knn.predict([[4.8,3.0,1.5,0.3]]))

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')