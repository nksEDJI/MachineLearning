import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier

#leitura dos dados
base_credit = pd.read_csv('credit_data.csv')
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92

#pega todos os valores usados para a machine learning e transforma em .values no X
X_credit = base_credit.iloc[:,1:4].values

#pega todos os valores da coluna target
Y_credit = base_credit.iloc[:, 4].values

#escalona os dados para ficarem equivalenets
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
print(X_credit[:,0].min(),X_credit[:,1].min(), X_credit[:,2].min())

#separando os dados de treinamento e teste
X_credit_train, X_credit_test, Y_credit_train, Y_credit_test = train_test_split(X_credit, Y_credit, test_size = 0.25, random_state= 0)

print(X_credit_train.shape)
print(Y_credit_train.shape)
print(X_credit_test.shape, Y_credit_test.shape)

with open('credit.pkl', mode = 'wb') as f:
    pickle.dump([X_credit_train,Y_credit_train, X_credit_test, Y_credit_test], f)

with open('credit.pkl', 'rb') as f:
    X_credit_train,Y_credit_train, X_credit_test, Y_credit_test = pickle.load(f)



plt.show()