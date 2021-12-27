import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

wine = datasets.load_wine()
print(dir(wine))
print(wine.data[0:2])
print(wine.feature_names)
print(wine.target_names)
print(wine.target[0:2])

df = pd.DataFrame(wine.data,columns=wine.feature_names)
print(df.head())

df['target'] = wine.target

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=100)

model = GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

mn = MultinomialNB()
mn.fit(X_train,y_train)
print(mn.score(X_test,y_test))