import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
print(data_read.describe())

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
    y_pred1 = knn.predict(inputs_train)         #faz a previs??o para os dados de treinamento com o".predict"
    error1.append(np.mean(targets_train!=y_pred1))  #mostra o valor do erro entre a previs??o de treinamento e os targets de treinamento
    y_pred2 = knn.predict(inputs_test)           #faz a previs??o para os dados de teste
    error2.append(np.mean(targets_test!=y_pred2))   #mostra o valor do err entre a previs??o de teste e os targets de teste
    #realiza a plotagem do grafico
fig, ax = plt.subplots(figsize=(16, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
plt.plot(range(1,15), error1, label = 'train')
plt.plot(range(1,15), error2, label = 'test')
plt.xlabel('k values', fontsize=18)
plt.ylabel('Error', fontsize=18)
plt.title('Econtrar "K"', fontsize = 18)
plt.legend()


#Realiza as defini????es do knn
X, Y = make_classification(n_samples= 250, n_features=10, n_informative=8, n_redundant=0, n_repeated=0, n_classes= 2, random_state=14)
#Passa os valores para X, Y que s??o as variaveis requeridas do algoritmo para suas defini????es

#--------------------------------------KNN--------------------------------#
#fazendo o classificador do algoritmo KNN
knn = KNeighborsClassifier(n_neighbors = 10) #utilizando a biblioteca do algoritmo fazemos as defini??oes para ele, decretando o valor de n

#treinando o algoritmo
knn.fit(inputs_train, targets_train) #com o ".fit" realizamos o treinamento, com X sendo os inputs de treinamento e Y os targets de treinamento

print("pontua????o knn: ", knn.score(inputs_test, targets_test))
#realizando as previsoes
previsoes = knn.predict(inputs_test) #com o ".predict" fazemos as previs??es com os inputs de teste

#Acuracia
print('Acuracia do KNN: ', metrics.accuracy_score(targets_test,previsoes)) #comparamos os targets de teste com a previs??o realizada para gerar o valor de acuracia do KNN

#--------------------------------------!KNN!--------------------------------#

#matris de confus??o
plt.figure(2)
plt.title('KNN')
ConfusionMatrix(knn).fit(inputs_train, targets_train) #treinamos a matriz de confus??o com os dados de treinamento
ConfusionMatrix(knn).score(inputs_test, targets_test) #realizamos a pontua????o da martiz de confus??o comparando os dados de treinamento com o de teste

#--------------------------------------Decision Tree--------------------------------#
decision_tree = DecisionTreeClassifier(random_state=0)
#treinar arvore de decis??o
decision_tree.fit(inputs_train, targets_train) #treinamos a arvore de decis??o com os dados de treinamento

#previs??es
previsoes = decision_tree.predict(inputs_test) #realizamos a previs??o dos valores de teste

#acuracia
accuracy = accuracy_score(targets_test, previsoes) #calculamos a pontua????o de acuracia dos targets de teste comparados a previs??o
print('Acuracia da decision tree: ', accuracy)


#--------------------------------------Decision Tree--------------------------------#

#matris de confus??o
plt.figure(3)
plt.title('Decision Tree')
ConfusionMatrix(decision_tree).fit(inputs_train, targets_train) #treinamos a matriz de confus??o para os dados de treino
ConfusionMatrix(decision_tree).score(inputs_test, targets_test) #pegamos a pontua????o comparando os dados do treinamento com os dados de teste

#dados de precis??o
print(classification_report(targets_test, previsoes)) #mosrta todos os valores adquiridos em todos os testtets





plt.show()
