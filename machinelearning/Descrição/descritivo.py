import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics

#leitura dos dados
base_credit = pd.read_csv('credit_data.csv')

#primeiros registros .head()
print(base_credit.head())

#ultimos registros .tail()
print(base_credit.tail())

#retorna todos os valores encontrados
print(base_credit.describe())

#retorna valores existente na base de dados
print(np.unique(base_credit['default'], return_counts = True ))
#com seus respectivos valores

#constroi grafico entre as variaveis
sns.countplot(x = base_credit['default'])

#mostra todos os valores existentes em um grafico
plt.hist(x = base_credit['income'])

#graficamente mostra os valores em formato iris
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default') 
#color diferencia os True dos False

#show() em uma variavel
grafico.show()

#localiza valores com o .loc
print(base_credit.loc[base_credit['age'] < 0])
#ou
print(base_credit[base_credit['age'] < 0])

#apaga valores inconsistentes do data frame
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)

#preencher com a media
base_credit.mean()
#media só de uma coluna
base_credit['age'].mean()
#media retirando os erros
base_credit['age'][base_credit['age'] > 0].mean()
#substituindo valores pela media
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92

#pega todos os valores usados para a machine learning e transforma em .values no X
X_credit = base_credit.iloc[:,1:4].values

#pega todos os valores da coluna target
Y_credit = base_credit.iloc[:, 4].values

#escalona os dados para ficarem equivalenets
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
X_credit[:,0].min(),X_credit[:,1].min(), X_credit[:,2].min()
X_credit[:,0].max(),X_credit[:,1].max(), X_credit[:,2].max()

#ver o tipo da variavel
type()

#separando os dados de treinamento e teste
X_credit_train, X_credit_test, Y_credit_train, Y_credit_test = train_test_split(X_credit, Y_credit, test_size = 0.25, random_state= 0)

print(X_credit_train.shape)
print(Y_credit_train.shape)
print(X_credit_test.shape, Y_credit_test.shape)

#salvar todos os dados de treinamento e teste separado
with open('credit.pkl', mode = 'wb') as f:
    pickle.dumb([X_credit_train,Y_credit_train, X_credit_test, Y_credit_test], f)

#abre os dados salvos 
with open('credit.pkl', 'rb') as f:
    X_credit_train,Y_credit_train, X_credit_test, Y_credit_test = pickle.load(f)



#mostra todos os graficos
plt.show()