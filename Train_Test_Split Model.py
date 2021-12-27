import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df=pd.read_csv(r"C:/Users/USER\Desktop/VS Code Programs/Python Programs/Machine Learning Programs/Car_prices.csv")
print(df.head())

#Kilometers covered vs selling price graph:-
plt.xlabel("Kilometers Covered")
plt.ylabel(" Selling Price")
plt.scatter(df["Km_Covered"],df["Price"])
plt.show()

#Age vs Selling price:-
plt.xlabel("Age")
plt.ylabel(" Selling Price")
plt.scatter(df["Age"],df["Price"])
plt.show()

x=df[["Km_Covered","Age"]]
y=df["Price"]
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(len(x_train))
print(len(x_test))

reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
print(reg.predict(x_test))
print(reg.score(x_test,y_test))