import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px

#Masculino = 1, Feminino = 0


#Lendo o arquivo
data_read = pd.read_csv('ChildrenData.csv')

#Retirando os dados que não serão usados
PropofolAdministration = data_read.drop(columns= ['Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15',]) #retira dados errados

#descrição dos dados
data_read.describe() #ve a descrição de tudo que tem nos dados

#definindo inputs e targets
inputs = data_read.iloc[0:46,6:10].values #pega todos os valores das colunas 6 a 9 para usar como inputs para treinamento e teste, utilizando o ".iloc" para selecionar os dados e o '.values' para pegar seus valores
targets = data_read.iloc[0:46,1:6].values #pega todos os valores da coluna 1 a 5 para usar como targets de resposta para o treinamento e teste


#escalonando os inputs para ficarem com valores equivalentes
inputs = StandardScaler().fit_transform(inputs) #transforma todos os valores dos inputs para numeros equivalente em decimais com o ".fit_transorm", o que aumenta a acertividade

#setando treinamento e teste
inputs_train, inputs_test, targets_train, targets_test  = train_test_split(inputs, targets, test_size= 0.5, random_state= 1)
#ao utilizar o 'train_test_split' o comando retorna os valores de treino e teste separando igualmente e aleatoriamente a quantidade de dados para treino e para teste

#analisar tamanho dos dados
inputs_train.shape, targets_train.shape
inputs_test.shape, targets_test.shape

#-------------------------------Decision Tree----------------------------#
#Realizando a regressão
TreeRegressor = DecisionTreeRegressor() #realiza a regressão dos dados pela arvore de decisão

#Treinando
TreeRegressor.fit(inputs_train, targets_train) #treina os dados de treinamento pelo medelo de regressão por arvore de decisões

#Pontuação
print("pontuação decision tree: ", TreeRegressor.score(inputs_test, targets_test)) #faz a pontuação comparando os valores de teste

#Previsoes
DecisionTree_previsoes = TreeRegressor.predict(inputs_test) #faz a previsão das respostas de target para os inpus de teste

#erro absoluto medio (MAE)
print("MAE decision tree: ",mean_absolute_error(targets_test, DecisionTree_previsoes)) #calcula o erro absoluto medio comparando as previsões com os targets de teste

#-------------------------------!/Decision Tree/!----------------------------#


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
simpe_regressor.fit(inputs_train, targets_train)

#pontuação
print("pontuação regressão linear simples: ",simpe_regressor.score(inputs_test, targets_test))

#previsões
simple_regressor_predict = simpe_regressor.predict(inputs_test)

#erro absoluto medio (MAE)
print("MAE regressão linear simples: ",mean_absolute_error(targets_test, simple_regressor_predict)) #calcula o erro absoluto medio comparando as previsões com os targets de teste

#plot Td
fig, ax = plt.subplots(figsize=(16, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
plt.plot(simple_regressor_predict[:,0], label = 'Regressão simples')
plt.plot(DecisionTree_previsoes[:,0], label = 'Decision Tree')
plt.plot(RandomForest_previsoes[:,0], label = 'Random Forest')
plt.plot(targets_test[:,0], label = 'Targets')
plt.title("Td - previsões X teste", fontsize=18)
plt.xlabel('Paciente', fontsize=18)
plt.ylabel('Valor', fontsize=18)
plt.legend()

#plot k
fig, ax = plt.subplots(figsize=(16, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
plt.plot(simple_regressor_predict[:,1], label = 'Regressão simples')
plt.plot(DecisionTree_previsoes[:,1], label = 'Decision Tree')
plt.plot(RandomForest_previsoes[:,1], label = 'Random Forest')
plt.plot(targets_test[:,1], label = 'Targets')
plt.title("K - previsões X teste", fontsize=18)
plt.xlabel('Paciente', fontsize=18)
plt.ylabel('Valor', fontsize=18)
plt.legend()

#plot E_50
fig, ax = plt.subplots(figsize=(16, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
plt.plot(simple_regressor_predict[:,2], label = 'Regressão simples')
plt.plot(DecisionTree_previsoes[:,2], label = 'Decision Tree')
plt.plot(RandomForest_previsoes[:,2], label = 'Random Forest')
plt.plot(targets_test[:,2], label = 'Targets')
plt.title("E_50 - previsões X teste", fontsize=18)
plt.xlabel('Paciente', fontsize=18)
plt.ylabel('Valor', fontsize=18)
plt.legend()

#plot E_0
fig, ax = plt.subplots(figsize=(16, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
plt.plot(simple_regressor_predict[:,3], label = 'Regressão simples')
plt.plot(DecisionTree_previsoes[:,3], label = 'Decision Tree')
plt.plot(RandomForest_previsoes[:,3], label = 'Random Forest')
plt.plot(targets_test[:,3], label = 'Targets')
plt.title("E_0 - previsões X teste", fontsize=18)
plt.xlabel('Paciente', fontsize=18)
plt.ylabel('Valor', fontsize=18)
plt.legend()

#plot Gamma
fig, ax = plt.subplots(figsize=(16, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
plt.plot(simple_regressor_predict[:,4], label = 'Regressão simples')
plt.plot(DecisionTree_previsoes[:,4], label = 'Decision Tree')
plt.plot(RandomForest_previsoes[:,4], label = 'Random Forest')
plt.plot(targets_test[:,4], label = 'Targets')
plt.title("Gamma - previsões X teste", fontsize=18)
plt.xlabel('Paciente', fontsize=18)
plt.ylabel('Valor', fontsize=18)
plt.legend()


plt.show()