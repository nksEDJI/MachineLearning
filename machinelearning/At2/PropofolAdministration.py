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
from yellowbrick.regressor import ResidualsPlot
from sklearn.metrics import mean_absolute_error
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.cluster import SilhouetteVisualizer

#Masculino = 1, Feminino = 0


#Lendo o arquivo
PropofolAdministration = pd.read_csv('ChildrenData.csv')

#Retirando os dados que não serão usados
PropofolAdministration = PropofolAdministration.drop(columns= ['Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15',])

#descrição dos dados
print(PropofolAdministration.describe())

#definindo inputs e targets
inputs = PropofolAdministration.iloc[0:46,6:10].values
targets = PropofolAdministration.iloc[0:46,1:6].values


#Deixa todos os inputs igualados em valores
sc = StandardScaler()
inputs = sc.fit_transform(inputs)

#setando treino e teste
inputs_train, inputs_test, targets_train, targets_test  = train_test_split(inputs, targets, test_size= 0.5, random_state= 1)

#analisar tamanho dos dados
print(inputs_train.shape, targets_train.shape)
print(inputs_test.shape, targets_test.shape )

#-------------------------------Decision Tree----------------------------#
#Realizando a regressão
TreeRegressor = DecisionTreeRegressor()

#Treinando
TreeRegressor.fit(inputs_train, targets_train)

#Pontuação
print(TreeRegressor.score(inputs_test, targets_test))

#Previsoes
DecisionTree_previsoes = TreeRegressor.predict(inputs_test)
print(DecisionTree_previsoes)

#erro absoluto medio (MAE)
print(mean_absolute_error(targets_test, DecisionTree_previsoes))

visualizer = SilhouetteVisualizer(TreeRegressor, colors='yellowbrick')

visualizer.fit(inputs_train, targets_train)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


#-------------------------------Random Forest----------------------------#
#Regressão
RandForestRegressor = RandomForestRegressor(n_estimators=100)

#treinamento
RandForestRegressor.fit(inputs_train, targets_train)

#Pontuação
RandForestRegressor.score(inputs_test, targets_test)

#Previsoes
RandomForest_previsoes = RandForestRegressor.predict(inputs_test)
print(RandomForest_previsoes)

#erro absoluto medio (MAE)
print(mean_absolute_error(targets_test, RandomForest_previsoes))

