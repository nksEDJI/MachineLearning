#importing necessary libraries
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn import preprocessing
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import plotly.express as px
import plotly.graph_objects as go
import warnings
from sklearn.linear_model import LogisticRegression
import optuna
from generate_data import *
from collections import Counter

warnings.filterwarnings("ignore")


set1 = pd.read_csv("machinelearning/set1_timefeatures.csv")
set2 = pd.read_csv("machinelearning/set2_timefeatures.csv")
set3 = pd.read_csv("machinelearning/set3_timefeatures.csv")

set1 = set1.rename(columns={'Unnamed: 0':'time'})
set1.set_index('time')
print(set1.describe())
set2 = set2.rename(columns={'Unnamed: 0':'time'}).set_index('time')
set3 = set3.rename(columns={'Unnamed: 0':'time'}).set_index('time')

time_features_list = ["mean","std","p2p", "impulse"]
bearings_xy = [["B"+str(n)+"_" for n in range(1,5)] for o in ['x','y'] ] 
#print(bearings_xy)
for tf in time_features_list:
    fig = plt.figure()
    # Divide the figure into a 1x4 grid, and give me the first section
    ax1 = fig.add_subplot(141)
    # Divide the figure into a 1x4 grid, and give me the second section
    ax2 = fig.add_subplot(142)
    #...so on
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    axes = [ax1,ax2,ax3, ax4]
    
    for i in range(4):
        col = bearings_xy[0][i]+tf
        set3[col].plot(figsize = (36,8), title="Bearing{} ".format(i+1)+tf , legend = True, ax=axes[i])
        col = bearings_xy[1][i]+tf
        set3[col].plot(figsize = (36,8) , legend = True, ax=axes[i])
        axes[i].set(xlabel="time", ylabel="value")
    plt.savefig(tf)


#Health Status labels are added according to following dictionary
B1 ={
    "early" : ["2003-10-22 12:06:24" , "2003-10-23 09:14:13"],
    "suspect" : ["2003-10-23 09:24:13" , "2003-11-08 12:11:44"],
    "normal" : ["2003-11-08 12:21:44" , "2003-11-19 21:06:07"],
    "suspect_1" : ["2003-11-19 21:16:07" , "2003-11-24 20:47:32"],
    "imminent_failure" : ["2003-11-24 20:57:32","2003-11-25 23:39:56"]
}
B2 = {
    "early" : ["2003-10-22 12:06:24" , "2003-11-01 21:41:44"],
    "normal" : ["2003-11-01 21:51:44" , "2003-11-24 01:01:24"],
    "suspect" : ["2003-11-24 01:11:24" , "2003-11-25 10:47:32"],
    "imminient_failure" : ["2003-11-25 10:57:32" , "2003-11-25 23:39:56"]
}

B3 = {
    "early" : ["2003-10-22 12:06:24" , "2003-11-01 21:41:44"],
    "normal" : ["2003-11-01 21:51:44" , "2003-11-22 09:16:56"],
    "suspect" : ["2003-11-22 09:26:56" , "2003-11-25 10:47:32"],
    "Inner_race_failure" : ["2003-11-25 10:57:32" , "2003-11-25 23:39:56"]
}

B4 = {
    "early" : ["2003-10-22 12:06:24" , "2003-10-29 21:39:46"],
    "normal" : ["2003-10-29 21:49:46" , "2003-11-15 05:08:46"],
    "suspect" : ["2003-11-15 05:18:46" , "2003-11-18 19:12:30"],
    "Rolling_element_failure" : ["2003-11-19 09:06:09" , "2003-11-22 17:36:56"],
    "Stage_two_failure" : ["2003-11-22 17:46:56" , "2003-11-25 23:39:56"]
}

B1_state = list()
B2_state = list()
B3_state = list()
B4_state = list()
cnt = 0

for row in set1["time"]:
    cnt += 1
    # B1
    if cnt<=151:
        B1_state.append("early")
    if 151 < cnt <=600:
        B1_state.append("suspect")
    if 600 < cnt <=1499:
        B1_state.append("normal")
    if 1499 < cnt <=2098:
        B1_state.append("suspect")
    if 2098 < cnt <= 2156:
        B1_state.append("imminent_failure")
    #B2
    if cnt<=500:
        B2_state.append("early")
    if 500 < cnt <=2000:
        B2_state.append("normal")
    if 2000 < cnt <=2120:
        B2_state.append("suspect")
    if 2120< cnt <=2156:
        B2_state.append("imminet_failure")

    #B3
    if cnt<=500:
        B3_state.append("early")
    if 500 < cnt <= 1790:
        B3_state.append("normal")
    if 1790 < cnt <=2120:
        B3_state.append("suspect")
    if 2120 < cnt <=2156:
        B3_state.append("Inner_race_failure")
    #B4
    if cnt<=200:
        B4_state.append("early")
    if 200 < cnt <=1000:
        B4_state.append("normal")
    if 1000 < cnt <= 1435:
        B4_state.append("suspect")
    if 1435 < cnt <=1840:
        B4_state.append("Inner_race_failure")
    if 1840 < cnt <=2156:
        B4_state.append("Stage_two_failure")

#controlling the counts


set1["B1_state"] = B1_state
set1["B2_state"] = B2_state
set1["B3_state"] = B3_state
set1["B4_state"] = B4_state
'''set2["B1_state"] = B1_state
set3['B1_state'] = B1_state'''

print(set1.head())
B1_cols = [col for col in set1.columns if "B1" in col]
B2_cols = [col for col in set1.columns if "B2" in col]
B3_cols = [col for col in set1.columns if "B3" in col]
B4_cols = [col for col in set1.columns if "B4" in col]
'''set2_B1_cols = [col for col in set2.columns if "B1" in col]
set3_B1_cols = [col for col in set3.columns if "B1" in col]'''

B1 = set1[B1_cols]
B2 = set1[B2_cols]
B3 = set1[B3_cols]
B4 = set1[B4_cols]

'''set2_B1 = set2[set2_B1_cols]
set3_B1 = set3[set3_B1_cols]'''
cols = ['Bx_mean','Bx_std','Bx_p2p','Bx_impulse',
        'By_mean','By_std','By_p2p', 'By_impulse',
        'class']
B1.columns = cols
B2.columns = cols
B3.columns = cols
B4.columns = cols

'''set2_B1.columns = cols
set3_B1.columns = cols'''




final_data = pd.concat([B1,B2,B3,B4], axis=0, ignore_index=True)
print(final_data.describe())

'''----------------------TREINAMENTO---------------------------------'''

X = final_data.copy()
y = X.pop("class")
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =1)
#X_x_train -> inputs_train
#y_train -> targets_train
#X_x_test -> inputs_test
#y_test -> targets_test



x_axis_cols = ["Bx_"+tf for tf in time_features_list]
print(x_axis_cols)
X_x = X.copy()
X_x = X[x_axis_cols]
cols = ['B_mean','B_std','B_p2p','B_impulse']
X_x.columns = cols
inputs_train, inputs_test, targets_train, targets_test = train_test_split(X_x, y, test_size = 0.3, random_state =1)
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
fig, ax = plt.subplots(figsize=(16, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)
plt.figure('casaquistao',figsize=(16,8))
plt.plot(range(1,15), error1, label = 'train')
plt.plot(range(1,15), error2, label = 'test')
plt.xlabel('k values', fontsize=18)
plt.ylabel('Error', fontsize=18)
plt.title('Econtrar "K"', fontsize = 18)
plt.legend()

names = ["KNN", 'Support Vector Classification',
         "Decision Tree", "Random Forest", "Neural Network", "AdaBoost",
         "Naive Bayes", "Quadratic Discriminant Analysis","XGBoost"]

'''names = ["Nearest Neighbors", 
         "Decision Tree", "Random Forest", "XGBoost"]
'''
'''classifiers = [
    KNeighborsClassifier(n_neighbors = 14),
    DecisionTreeClassifier(random_state=1, min_samples_split=5, min_samples_leaf=5),
    RandomForestClassifier( n_estimators=100, random_state=1),
    xgb.XGBClassifier(random_state=1)
    ]'''
classifiers = [    
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    xgb.XGBClassifier(),
]

for name, clf in zip(names,classifiers):
    print("training "+name+" ...")
    clf.fit(inputs_train,targets_train)
    score = clf.score(inputs_test,targets_test)
    print('Score of '+name+' is: '+str(score))
    #iterate over classifiers

    

final_model_xgb = xgb.XGBClassifier(random_state=1)
final_model_xgb.fit(inputs_train, targets_train)
preds_xgb = final_model_xgb.predict(inputs_test)
print(accuracy_score(targets_test, preds_xgb))

final_model_DecisionTree = DecisionTreeClassifier(random_state=1, min_samples_split=5, min_samples_leaf=5)
final_model_DecisionTree.fit(inputs_train, targets_train)
preds_DecisionTree = final_model_DecisionTree.predict(inputs_test) 
print(accuracy_score(targets_test, preds_DecisionTree))

final_model_RandomForest = RandomForestClassifier( n_estimators=100, random_state=1)
final_model_RandomForest.fit(inputs_train, targets_train)
preds_RandomForest = final_model_RandomForest.predict(inputs_test)
print(accuracy_score(targets_test, preds_RandomForest))
#------------------------------------------------------------------------------------

B1_cols = [col for col in set2.columns if "B1" in col]
B2_cols = [col for col in set2.columns if "B2" in col]
B3_cols = [col for col in set2.columns if "B3" in col]
B4_cols = [col for col in set2.columns if "B4" in col]

set2_B1 = set2[B1_cols]
set2_B2 = set2[B2_cols]
set2_B3 = set2[B3_cols]
set2_B4 = set2[B4_cols]

set3_B1 = set3[B1_cols]
set3_B2 = set3[B2_cols]
set3_B3 = set3[B3_cols]
set3_B4 = set3[B4_cols]

set2_B1.columns = cols
set2_B2.columns = cols
set2_B3.columns = cols
set2_B4.columns = cols
set3_B1.columns = cols
set3_B2.columns = cols
set3_B3.columns = cols
set3_B4.columns = cols

# HERE number and dataset of bearing can be changed !!!
bearing = set2_B1

#predicting state of bearing with final_model
preds = final_model_xgb.predict(bearing)
preds = le.inverse_transform(preds)
#inserting prediction and time to the dataframe
bearing.insert(4,'state',preds)
bearing.insert(5, 'time',bearing.index)

for tf in time_features_list:
    col = "B_{}".format(tf)
    print(col)
    fig=go.Figure((go.Scatter(x=bearing['time'], y=bearing[col],
                             mode='lines',
                             line=dict(color='rgba(0,0,220,0.8)'))))
    fig.add_traces(px.scatter(bearing, x='time', y=col, color='state').data)
    fig.update_layout(template='plotly_dark',title=col, xaxis_title='time', yaxis_title=col)
    fig.update_xaxes(showgrid=False)
    fig.show()
    
bearing = set3_B1

#predicting state of bearing with final_model
preds = final_model_xgb.predict(bearing)
preds = le.inverse_transform(preds)
#inserting prediction and time to the dataframe
bearing.insert(4,'state',preds)
bearing.insert(5, 'time',bearing.index)

for tf in time_features_list:
    col = "B_{}".format(tf)
    print(col)
    fig=go.Figure((go.Scatter(x=bearing['time'], y=bearing[col],
                             mode='lines',
                             line=dict(color='rgba(0,0,220,0.8)'))))
    fig.add_traces(px.scatter(bearing, x='time', y=col, color='state').data)
    fig.update_layout(template='plotly_dark',title=col, xaxis_title='time', yaxis_title=col)
    fig.update_xaxes(showgrid=False)
    fig.show()


plt.show()