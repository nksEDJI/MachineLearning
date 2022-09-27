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
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

#ler dados
data = pd.read_csv('inputs.csv')

#ver os parametros dos dados
print(data.describe())

#separando inputs e targets
inputs = data.iloc[:,0:10].values
targets = data.iloc[:,10].values

#escalonando os inputs para ficarem com valores equivalentes
sc = StandardScaler()
inputs = sc.fit_transform(inputs)

#setando treinamento e teste
inputs_train, inputs_test, targets_train, targets_test  = train_test_split(inputs, targets, test_size= 0.3, random_state= 1)

#analisar tamanho dos dados
print(inputs_train.shape, targets_train.shape)
print(inputs_test.shape, targets_test.shape )

error1 = []
error2 = []

#descobrindo o melhor "k" para o algoritmo

for k in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(inputs_train, targets_train)
    y_pred1 = knn.predict(inputs_train)
    error1.append(np.mean(targets_train!=y_pred1))
    y_pred2 = knn.predict(inputs_test)
    error2.append(np.mean(targets_test!=y_pred2))
plt.figure(1, figsize=(18,8))
plt.plot(range(1,15), error1, label = 'train')
plt.plot(range(1,15), error2, label = 'test')
plt.xlabel('k values')
plt.ylabel('Error')
plt.title('Econtrar "K"')
plt.legend()


#Realiza as definições do knn
X, Y = make_classification(n_samples= 250, n_features=10, n_informative=8, n_redundant=0, n_repeated=0, n_classes= 2, random_state=14)

#--------------------------------------Algoritmo--------------------------------
#fazendo o classificador do algoritmo KNN
knn = KNeighborsClassifier(n_neighbors = 10)

#treinando o algoritmo
knn.fit(inputs_train, targets_train)

#realizando as previsoes
previsoes = knn.predict(inputs_test)

#Acuracia
print('Acuracia do KNN: ', metrics.accuracy_score(targets_test,previsoes))

#--------------------------------------!/Algoritmo/!--------------------------------

#matris de confusão
plt.figure(2)
plt.title('KNN')
cm = ConfusionMatrix(knn)
cm.fit(inputs_train, targets_train)
cm.score(inputs_test, targets_test)

#--------------------------------------Algoritmo--------------------------------
#treinar arvore de decisão
DecisionTree = DecisionTreeClassifier(random_state=0)
DecisionTree.fit(inputs_train, targets_train)

#previsões
previsoes = DecisionTree.predict(inputs_test)

#acuracia
accuracy = accuracy_score(targets_test, previsoes)
print('Acuracia da decision tree: ', accuracy)

#matris de confusão
plt.figure(3)
plt.title('Decision Tree')
cm = ConfusionMatrix(DecisionTree)
cm.fit(inputs_train, targets_train)
cm.score(inputs_test, targets_test)

#dados de precisão
print(classification_report(targets_test, previsoes))





plt.show()
