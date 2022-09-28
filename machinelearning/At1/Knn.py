import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.tree import DecisionTreeClassifier

data_read = pd.read_csv('inputs.csv')

#ver os parametros dos dados
data_read.describe()

#separando inputs e targets
inputs = data_read.iloc[:,0:10].values #pega todos os valores das colunas 0 a 9 para usar como inputs para treinamento e teste, utilizando o ".iloc" para selecionar os dados e o '.values' para pegar seus valores
targets = data_read.iloc[:,10].values #pega todos os valores da coluna 10 para usar como targets de resposta para o treinamento e teste

#escalonando os inputs para ficarem com valores equivalentes
inputs = StandardScaler().fit_transform(inputs) #transforma todos os valores dos inputs para numeros equivalente em decimais com o ".fit_transorm", o que aumenta a acertividade

#setando treinamento e teste
inputs_train, inputs_test, targets_train, targets_test  = train_test_split(inputs, targets, test_size= 0.3, random_state= 1)
#ao utilizar o 'train_test_split' o comando retorna os valores de treino e teste separando igualmente e aleatoriamente a quantidade de dados para treino e para teste


#analisar tamanho dos dados
print(inputs_train.shape, targets_train.shape) #utilizando o ".shape" conseguimos ver quantos valores temos e quantas colunas para analisar qual algoritmo usar
print(inputs_test.shape, targets_test.shape )



#descobrindo o melhor "k" para o algoritmo
error1 = []
error2 = []

for k in range(1,15): #utilizamos um for de 15 variaveis para testar o "k" do knn de 0 a 15 e encontrar o meelhor k
    knn = KNeighborsClassifier(n_neighbors=k)       #modifica o valor de "k" no algoritmo para ter a sua resposta
    knn.fit(inputs_train, targets_train)        #realiza o treinamento usando os inputs e targets de treino com o ".fit"
    y_pred1 = knn.predict(inputs_train)         #faz a previsão para os dados de treinamento com o".predict"
    error1.append(np.mean(targets_train!=y_pred1))  #mostra o valor do erro entre a previsão de treinamento e os targets de treinamento
    y_pred2 = knn.predict(inputs_test)           #faz a previsão para os dados de teste
    error2.append(np.mean(targets_test!=y_pred2))   #mostra o valor do err entre a previsão de teste e os targets de teste
    #realiza a plotagem do grafico
plt.figure(1, figsize=(18,8))
plt.plot(range(1,15), error1, label = 'train')
plt.plot(range(1,15), error2, label = 'test')
plt.xlabel('k values')
plt.ylabel('Error')
plt.title('Econtrar "K"')
plt.legend()


#Realiza as definições do knn
X, Y = make_classification(n_samples= 250, n_features=10, n_informative=8, n_redundant=0, n_repeated=0, n_classes= 2, random_state=14)
#Passa os valores para X, Y que são as variaveis requeridas do algoritmo para suas definições

#--------------------------------------KNN--------------------------------#
#fazendo o classificador do algoritmo KNN
knn = KNeighborsClassifier(n_neighbors = 10) #utilizando a biblioteca do algoritmo fazemos as definiçoes para ele, decretando o valor de n

#treinando o algoritmo
knn.fit(inputs_train, targets_train) #com o ".fit" realizamos o treinamento, com X sendo os inputs de treinamento e Y os targets de treinamento

#realizando as previsoes
previsoes = knn.predict(inputs_test) #com o ".predict" fazemos as previsões com os inputs de teste

#Acuracia
print('Acuracia do KNN: ', metrics.accuracy_score(targets_test,previsoes)) #comparamos os targets de teste com a previsão realizada para gerar o valor de acuracia do KNN

#--------------------------------------!KNN!--------------------------------#

#matris de confusão
plt.figure(2)
plt.title('KNN')
ConfusionMatrix(knn).fit(inputs_train, targets_train) #treinamos a matriz de confusão com os dados de treinamento
ConfusionMatrix(knn).score(inputs_test, targets_test) #realizamos a pontuação da martiz de confusão comparando os dados de treinamento com o de teste

#--------------------------------------Decision Tree--------------------------------#
decision_tree = DecisionTreeClassifier(random_state=0)
#treinar arvore de decisão
decision_tree.fit(inputs_train, targets_train) #treinamos a arvore de decisão com os dados de treinamento

#previsões
previsoes = decision_tree.predict(inputs_test) #realizamos a previsão dos valores de teste

#acuracia
accuracy = accuracy_score(targets_test, previsoes) #calculamos a pontuação de acuracia dos targets de teste comparados a previsão
print('Acuracia da decision tree: ', accuracy)


#--------------------------------------Decision Tree--------------------------------#

#matris de confusão
plt.figure(3)
plt.title('Decision Tree')
ConfusionMatrix(decision_tree).fit(inputs_train, targets_train) #treinamos a matriz de confusão para os dados de treino
ConfusionMatrix(decision_tree).score(inputs_test, targets_test) #pegamos a pontuação comparando os dados do treinamento com os dados de teste

#dados de precisão
print(classification_report(targets_test, previsoes)) #mosrta todos os valores adquiridos em todos os testtets





plt.show()
