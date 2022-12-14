import pandas as pd

dataset_train=pd.read_csv("Book3.csv")
dataset_test=pd.read_csv("Book2.csv")
x_train=dataset_train.iloc[:,1:].values
y_train=dataset_train.iloc[:,0].values
x_test=dataset_test.iloc[:,1:].values
y_test=dataset_test.iloc[:,0].values

print(dataset_train['23'].value_counts())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=100000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score

acc= accuracy_score(y_test,y_pred)
print(acc)