# Importando bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# lendo csv e armazenando em um dataframe
dados = pd.read_csv('data.csv')
print(dados.head())

#verificar se existem valores NAN, ? ou dados faltantes
dados = dados.dropna(axis=1)

#excluir colunas irrelevantes
dados = dados.drop(columns=['id'])
print(dados.head())

#trocando o tipo do atributo diagnostico por um tipo numerico
dados['diagnosis'] = dados['diagnosis'].replace(['M', 'B'], [1, 0])
print(dados.head())

# Re-escala usando máximo e mínimo
dados = (dados - dados.min())/(dados.max()-dados.min())

#dividindo dados em atributos descritores e atributo de classe
X = dados.iloc[:,1:]
print(X.head())

y = dados.diagnosis
print(y.head())

# Dividir os dados em treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)#random_state=42
print(X_train.head())
print(y_train.head())

# Definindo algoritmo de aprendizado
# Rede Neural Multi-Layer Perceptron (MLP)
from sklearn.neural_network import MLPClassifier

#definindo modelo
classificador = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',max_iter=1000)

#treinando modelo
classificador.fit(X_train,y_train)

#realizando classificação
classificacao = classificador.predict(X_test)
print(classificacao)

# Avaliação do classificador
# Acurácia - taxa de acertos do classificador
# calculando acurácia
from sklearn.metrics import accuracy_score
acuracia = accuracy_score(y_test,classificacao)
print("Acuracia: ", round(acuracia,3))

# Precisão - taxa de instâncias classificadas como positivas que são realmente positivas
# calculando precisão
from sklearn.metrics import precision_score
precisao = precision_score(y_test,classificacao)
print("Precisao: ", round(precisao,3))

# Recall - taxa de instâncias positivas classificadas corretamente
#calculando recall (revocação)
from sklearn.metrics import recall_score
recall = recall_score(y_test,classificacao)
print("Recall: ", round(recall,3))

# F1-score - balanço entre precisão e recall
#calculando f1-score
from sklearn.metrics import f1_score
f1 = f1_score(y_test,classificacao)
print("F1-score: ", round(f1,3))

# Curva ROC - Representação gráfica do desempenho de um classificador binário
#plotando curva roc
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test,classificacao)

plt.plot(fpr,tpr,marker='.')
plt.title('Curva ROC')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiro Positivos')
plt.show()

# Área sob a curva (*Area under the curve - AUC)
#calculando area sob a curva ROC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test,classificacao)
print("Area sob a curva: ", round(auc,3))

# Validação cruzada
# avaliando modelo com cross validation
from sklearn.model_selection import cross_val_score
#define modelo
classificador = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',max_iter=1000)
#calculando os scores
scores = cross_val_score(classificador,X,y,cv=10)
print(scores)

print("Score da validacao cruzada: ", round(scores.mean(),3),round(scores.std(),3))

# Comparando MLP com Árvore de Decisão e Random Forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#criando árvore
arvore = DecisionTreeClassifier()

#calculando os scores
scores_arvore = cross_val_score(arvore,X,y,cv=10)

#criando random forest
floresta = RandomForestClassifier()

#calculando os scores
scores_floresta = cross_val_score(floresta,X,y,cv=10)

#criando rede neural
mlp = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',max_iter=1000)

#calculando os scores
scores_mlp = cross_val_score(mlp,X,y,cv=10)

print("Arvore de Decisao: ", round(scores_arvore.mean(),3),round(scores_arvore.std(),3))
print("Random Forest: ", round(scores.mean(),3),round(scores.std(),3))
print("MLP: ", round(scores_mlp.mean(),3),round(scores_mlp.std(),3))

# Gráfico para visualização dos dados
# correlaçao entre os atributos
corr = dados.corr()
print(corr)

#plotando coeficientes de correlação em um mapa de calor
sns.heatmap(corr,vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200))
plt.show()

#criando dataframe apenas para os nodulos benignos
benigno = dados[dados['diagnosis']==0].drop(columns=['diagnosis']).reset_index(drop=True)

#criando dataframe apenas para os nodulos benignos
maligno = dados[dados['diagnosis']==1].drop(columns=['diagnosis']).reset_index(drop=True)

#plotando boxplots dos dados das instâncias benignas
benigno.boxplot()
plt.title('Box plot dos atributos dos nódulos benignos')
plt.show()

#plotando boxplots dos dados das instâncias malignas
maligno.boxplot()
plt.title('Box plot dos atributos dos nódulos malignos')
plt.show()