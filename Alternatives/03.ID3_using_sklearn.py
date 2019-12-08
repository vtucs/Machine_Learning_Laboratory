import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

train_data=list(csv.reader(open('ds2.csv')))[1:]
test_data=['Rainy','Mild','High','True','NaN']
train_data.append(test_data)

train_data=np.array(train_data).transpose()
encoder=LabelEncoder()
for i in range(len(train_data)):
    data[i]=encoder.fit_transform(train_data[i])

x=train_data[:-1].transpose()
y=train_data[-1]
test_data=x[-1]

tree=DecisionTreeClassifier()
tree.fit(x[:-1],y[:-1])
pred=tree.predict([x[-1]])

if pred[0]=='2':
    print('Can play tennis')
elif pred[0]=='1':
    print('Cannot play tennis')    