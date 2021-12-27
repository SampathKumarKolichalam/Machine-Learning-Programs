#Binary Classification:-
from numpy import mod
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("Insurance_data.csv")
#print(df.head())
plt.scatter(df.Age,df.Insurances,marker="+",color="red")
#plt.show()

x_train,x_test,y_train,y_test=train_test_split(df[["Age"]],df.Insurances,test_size=0.1)
model=LogisticRegression()
model.fit(x_train,y_train)
y_predicted=model.predict(x_test)
print(model.predict_proba(x_test))
print(model.predict(x_test))
print(model.score(x_test,y_test))



#Multiclass Classification:-
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits=load_digits()
#print(dir(digits))
#print(digits.data[0])
#plt.gray()
#for i in range(5):
#plt.matshow(digits.images[i])

model=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)
model.fit(x_train,y_train)
#print(model.score(x_test,y_test))
#print(model.predict(digits.data[0:5]))

#Confusion Matrix:-
y_predicted=model.predict(x_test)
cm=confusion_matrix(y_test,y_predicted)
#print(cm)

#Heat Map:-
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()