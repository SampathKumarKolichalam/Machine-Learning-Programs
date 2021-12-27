import numpy as np
import pandas as pd
from word2number import w2n
from sklearn import linear_model

df=pd.read_csv(r"C:/Users/USER\Desktop/VS Code Programs/Python Programs/Machine Learning Programs/Hiring_Data.csv")
#print(df)

df.experience=df.experience.fillna("zero")
#print(df)

df.experience=df.experience.apply(w2n.word_to_num)
#print(df)

reg=linear_model.LinearRegression()
print(reg.predict([[2,9,6]]))
print(reg.predict([[12,10,10]]))
#print(reg.coef_)
#print(reg.intercept_)