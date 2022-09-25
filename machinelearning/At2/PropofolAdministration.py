import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from yellowbrick.regressor import ResidualsPlot


#Macho = 1, Femea = 0


#Lendo o arquivo
PropofolAdministration = pd.read_csv('ChildrenData.csv')



#Retirando os dados que não serão usados
PropofolAdministration = PropofolAdministration.drop(columns= ['Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15',])

#descrição dos dados
print(PropofolAdministration.describe())

#pega todos os valores usados para a machine learning e transforma em .values no X
inputs = PropofolAdministration.iloc[0:46,6:10].values

#pega todos os valores da coluna target
targets = PropofolAdministration.iloc[0:46,1:6].values

#Deixa todos os inputs igualados em valores
sc = StandardScaler()
inputs = sc.fit_transform(inputs)




#setando treino e teste
inputs_train, targets_test, targets_train, inputs_test = train_test_split(inputs, targets, test_size= 0.3, random_state= 1)

print(inputs_train.shape)
print(targets_train.shape)
print(targets_test.shape, inputs_test.shape)

regressor_multiplo = LinearRegression()
regressor_multiplo.fit(inputs_train, targets_train)
print(regressor_multiplo.intercept_)
print(regressor_multiplo.coef_)
print(len(regressor_multiplo.coef_))

score = regressor_multiplo.score(inputs_train, targets_train)
print(score)

score1 = regressor_multiplo.score(inputs_test, targets_test)
print(score1)



'''
DecisionTree = DecisionTreeClassifier(criterion='entrioy',random_state=0)
DecisionTree.fit(inputs_train, targets_train)'''