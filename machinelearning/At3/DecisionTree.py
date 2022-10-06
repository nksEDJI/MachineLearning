from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from sklearn.datasets import load_iris

iris = load_iris()
iris

#ler dados
WaterDistribuition = pd.read_csv('WaterDistribuition.csv')

#descrição dos dados
WaterDistribuition.describe()

#encontrar dados faltantes
print(WaterDistribuition.isnull().sum()) #soma os valores NaN dos dados

#preenchendo os dados faltantes com a media
WaterDistribuition['WaterTemperature'].fillna(WaterDistribuition['WaterTemperature'].mean(), inplace = True)
WaterDistribuition['CIO2_1'].fillna(WaterDistribuition['CIO2_1'].mean(), inplace = True)
WaterDistribuition['pH'].fillna(WaterDistribuition['pH'].mean(), inplace = True)
WaterDistribuition['Redox'].fillna(WaterDistribuition['Redox'].mean(), inplace = True)
WaterDistribuition['electro-conductividade'].fillna(WaterDistribuition['electro-conductividade'].mean(), inplace = True)
WaterDistribuition['turbiedade'].fillna(WaterDistribuition['turbiedade'].mean(), inplace = True)
WaterDistribuition['CIO2_2'].fillna(WaterDistribuition['CIO2_2'].mean(), inplace = True)
WaterDistribuition['FlowRate1'].fillna(WaterDistribuition['FlowRate1'].mean(), inplace = True)
WaterDistribuition['FlowRate2'].fillna(WaterDistribuition['FlowRate2'].mean(), inplace = True)


print(WaterDistribuition.describe())
#denominando inputs e targets
inputs = WaterDistribuition.iloc[:,0:9].values
targets = WaterDistribuition.iloc[:,9].values
print(inputs.shape)
print(targets.shape)
#escalonando dados
inputs = StandardScaler().fit_transform(inputs)


#setando treino e teste
inputs_train, inputs_test, targets_train, targets_test  = train_test_split(inputs, targets, test_size= 0.3, random_state= 1)


#analisar tamanho dos dados
print(inputs_train.shape, targets_train.shape)
print(inputs_test.shape, targets_test.shape )

#treinar arvore de decisão
DecisionTree = DecisionTreeClassifier(random_state=0)
DecisionTree.fit(inputs_train, targets_train)

#previsões
previsoes = DecisionTree.predict(inputs_test)

#acuracia
accuracy = accuracy_score(targets_test, previsoes)
print("Acuracia decision tree: ",accuracy)

#matris de confusão
plt.figure(1)
plt.title('Decision Tree')
cm = ConfusionMatrix(DecisionTree)
cm.fit(inputs_train, targets_train)
cm.score(inputs_test, targets_test)





#dados de precisão
print(classification_report(targets_test, previsoes))
#------------------------------Random forest---------------------#
random_forest_classifier = RandomForestClassifier(random_state = 0, n_estimators=100)
random_forest_classifier.fit(inputs_train, targets_train)
previsoes2 = random_forest_classifier.predict(inputs_test)
accuracy1 = accuracy_score(targets_test, previsoes2)
print("Acuracia random forest: ",accuracy1)

#matris de confusão
plt.figure(2)
plt.title('Random Forest')
cm = ConfusionMatrix(random_forest_classifier)
cm.fit(inputs_train, targets_train)
cm.score(inputs_test, targets_test)





plt.show()