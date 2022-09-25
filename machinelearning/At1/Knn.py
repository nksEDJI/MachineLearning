import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import make_classification

#ler inputs
inputs = pd.read_csv('inputs.csv')

print(inputs)
print(inputs.describe())

#quantidade de targets negativos e positivos
print(np.unique(inputs['targets'], return_counts = True))


plt.hist(x=inputs['A'])

#diminua!!!!
grafico = px.scatter_matrix(inputs, dimensions=['A','B','C'], color = 'targets')
grafico.show()

x_inputs = inputs.iloc[:,0:10].values
y_inputs = inputs.iloc[:,10].values


print('esse daqui',x_inputs)
print(y_inputs)

sc = StandardScaler()
x_inputs = sc.fit_transform(x_inputs)


x_inputs_train, x_inputs_test, y_inputs_train, y_inputs_test = train_test_split(x_inputs, y_inputs, test_size= 0.25, random_state= 1)

print(x_inputs_train.shape)
print( y_inputs_train.shape)
print(x_inputs_test.shape, y_inputs_test.shape)

X, Y = make_classification(n_samples= 250, n_features=10, n_informative=8, n_redundant=0, n_repeated=0, n_classes= 2, random_state=14)

print(X.shape)

error1 = []
error2 = []

for k in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_inputs_train, y_inputs_train)
    y_pred1 = knn.predict(x_inputs_train)
    error1.append(np.mean(y_inputs_train!=y_pred1))
    y_pred2 = knn.predict(x_inputs_test)
    error2.append(np.mean(y_inputs_test!=y_pred2))
plt.figure(figsize=(18,8))
plt.plot(range(1,15), error1, label = 'train')
plt.plot(range(1,15), error2, label = 'test')
plt.xlabel('k values')
plt.ylabel('Error')
plt.legend()


knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_inputs_train, y_inputs_train)
y_pred = knn.predict(x_inputs_test)
print(metrics.accuracy_score(y_inputs_test,y_pred))

plt.show()
