#Importing Libraries:-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
from sklearn.externals import joblib

df=pd.read_csv(r"C:/Users/USER\Desktop/VS Code Programs/Python Programs/Machine Learning Programs/House_Prices.csv")
print(df)

plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.area,df.price,color="red",marker="*")
plt.show()

reg=linear_model.LinearRegression()
reg.fit(df[["area"]],df.price)
print(reg.predict([[3300]]))
print(reg.coef_)
print(reg.intercept_)

#Using pickle Module/Library:-
with open("Model_Pickle","wb") as file:
    pickle.dump(reg,file)

with open("Model_Pickle","rb") as file:
    model=pickle.load(file)
print(model.predict([[5000]]))

#Using joblib Module/Library:-
joblib.dump(reg,"Model_joblib")
mjb=joblib.load("Model_joblib")
print(mjb.predict([[5000]]))


