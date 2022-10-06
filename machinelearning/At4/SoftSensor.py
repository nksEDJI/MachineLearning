import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
#ler dados
SoftSensor = pd.read_csv('Behavior.csv')

#Retirando os dados que não serão usados
SoftSensor = SoftSensor.drop(columns= ['Unnamed: 7','Unnamed: 8','Unnamed: 9'])

#descrição dos dados
SoftSensor.describe()

#denominando inputs e targets
inputs = SoftSensor.iloc[:,0:5].values
targets = SoftSensor.iloc[:,5:7].values

#escalonando dados
inputs = StandardScaler().fit_transform(inputs)

#setando treino e teste
inputs_train, inputs_test, targets_train, targets_test  = train_test_split(inputs, targets, test_size= 0.3, random_state= 1)

#analisar tamanho dos dados
print(inputs_train.shape, targets_train.shape)
print(inputs_test.shape, targets_test.shape )

#-------------------------------Decision Tree----------------------------#
#Realizando a regressão
TreeRegressor = DecisionTreeRegressor()

#Treinando
TreeRegressor.fit(inputs_train, targets_train)

#Pontuação
print("pontuação Decision Tree: ",TreeRegressor.score(inputs_test, targets_test))

#Previsoes
DecisionTree_previsoes = TreeRegressor.predict(inputs_test)

#erro absoluto medio (MAE)
print("MAE Decision Tree: ",mean_absolute_error(targets_test, DecisionTree_previsoes))

#-------------------------------Random Forest----------------------------#
#Regressão
random_forest_regressor = RandomForestRegressor(n_estimators=100) #realiza a regressão dos dados pela arvore aleatoria

#treinamento
random_forest_regressor.fit(inputs_train, targets_train) #treina os dados de treinamento pelo medelo de regressão por arvore aleatorias

#Pontuação
print("pontuação random forest: ",random_forest_regressor.score(inputs_test, targets_test)) #faz a pontuação comparando os valores de teste

#Previsoes
RandomForest_previsoes = random_forest_regressor.predict(inputs_test) #faz a previsão das respostas de target para os inpus de teste


#erro absoluto medio (MAE)
print("MAE random forest: ", mean_absolute_error(targets_test, RandomForest_previsoes)) #calcula o erro absoluto medio comparando as previsões com os targets de teste

#-------------------------------!/Random Forest/!----------------------------#

#-------------------------------Regressão linear simples----------------------------#
#treinamento
simpe_regressor = LinearRegression()
LinearRegression().fit(inputs_train, targets_train)

#pontuação
print("pontuação regressão linear simples: ",LinearRegression().fit(inputs_train, targets_train).score(inputs_test, targets_test))

#previsões
simple_regressor_predict = LinearRegression().fit(inputs_train, targets_train).predict(inputs_test)

#erro absoluto medio (MAE)
print("MAE regressão linear simples: ",mean_absolute_error(targets_test, simple_regressor_predict)) #calcula o erro absoluto medio comparando as previsões com os targets de teste

#plot Td
fig, ax = plt.subplots(figsize=(16, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
plt.plot(targets_test[0:25,0], label = 'Targets', color = 'red')
plt.plot(DecisionTree_previsoes[0:25,0], label = 'Decision Tree', color = 'black')
plt.plot(RandomForest_previsoes[0:25,0], label = 'Random Forest', color = 'yellow')
plt.title("hot - previsões X teste", fontsize=18)
plt.ylabel('Temperatura', fontsize=18)
plt.legend(fontsize=18)
plt.savefig('6')
#plot k
fig, ax = plt.subplots(figsize=(16, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
plt.plot(targets_test[0:25,1], label = 'Targets', color = 'red')
plt.plot(DecisionTree_previsoes[0:25,1], label = 'Decision Tree', color = 'black')
plt.plot(RandomForest_previsoes[0:25,1], label = 'Random Forest', color = 'yellow')
plt.title("rad - previsões X teste", fontsize=18)
plt.ylabel('Temperatura', fontsize=18)
plt.legend(fontsize=18)
plt.savefig('7')


plt.show()


