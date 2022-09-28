from tkinter import W
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree


#ler dados
WaterDistribuition = pd.read_csv('WaterDistribuition.csv')

#descrição dos dados
WaterDistribuition.describe()

#quantidade de targets negativos e positivos
np.unique(WaterDistribuition['Targets'], return_counts = True)

#encontrar dados faltantes
WaterDistribuition.isnull().sum()

#ver os dados faltantes
WaterDistribuition.loc[pd.isnull(WaterDistribuition ['pH'])]

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



#confirmar dados faltantes preenchidos
WaterDistribuition.isnull().sum()



#denominando inputs e targets
inputs = WaterDistribuition.iloc[:,0:9].values
targets = WaterDistribuition.iloc[:,9].values

#escalonando dados
sc = StandardScaler()
inputs = sc.fit_transform(inputs)


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

#mostrando
'''plt.figure(2)
tree.plot_tree(DecisionTree)
previsores = ['WaterTemperature', 'CIO2_1', 'pH', 'pH', 'electro-conductividade', 'turbiedade', 'CIO2_2', 'FlowRate1', 'FlowRate2']
fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (20,20))
tree.plot_tree(DecisionTree, feature_names = previsoes, class_names=['0','1'], filled=True)'''


plt.show()