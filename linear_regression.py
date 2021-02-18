import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

dados = pd.read_csv('dataset.csv', sep=';')
dados.head()

dados.shape
dados.describe().round(2)

dados.corr().round(4) 

#Distribuição de frequências da variável dependente
ax = sns.distplot(dados['Valor'])
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize=20)
ax.set_xlabel('Preço dos Imóveis (R$)', fontsize=16)
ax

#Gráficos de dispersão entre as variáveis do dataset
ax = sns.pairplot(dados, y_vars="Valor", x_vars=["Area", "Dist_Praia", "Dist_Farmacia"], height=5)
ax.fig.suptitle('Dispersão entre as Variáveis', fontsize=20, y=1.05)
ax

ax = sns.pairplot(dados, y_vars="Valor", x_vars=["Area", "Dist_Praia", "Dist_Farmacia"], kind='reg', height=5)
ax.fig.suptitle('Dispersão entre as Variáveis', fontsize=20, y=1.05)
ax

dados['log_Valor'] = np.log(dados['Valor'])
dados['log_Area'] = np.log(dados['Area'])
dados['log_Dist_Praia'] = np.log(dados['Dist_Praia'] + 1)
dados['log_Dist_Farmacia'] = np.log(dados['Dist_Farmacia'] + 1)

dados.head()

#Distribuição de frequências da variável dependente transformada (y)
ax = sns.distplot(dados['log_Valor'])
ax.figure.set_size_inches(12, 6)
ax.set_title('Distribuição de Frequências', fontsize=20)
ax.set_xlabel('log do Preço dos Imóveis', fontsize=16)
ax

#Gráficos de dispersão entre as variáveis transformadas do dataset
ax = sns.pairplot(dados, y_vars="log_Valor", x_vars=["log_Area", "log_Dist_Praia", "log_Dist_Farmacia"], kind='reg', height=5)
ax.fig.suptitle('Dispersão entre as Variáveis Transformadas', fontsize=20, y=1.05)
ax

y = dados['log_Valor']
x = dados[['log_Area','log_Dist_Praia','log_Dist_Farmacia']]

#Criando os datasets de treino e de teste
x_train, x_test, y_train, x_test = train_test_split(x, y, test_size = 0.2, random_state = 2811)

x_train_com_constante = sm.add_constant(x_train)

#Avaliando as estatísticas de teste do modelo
modelo_statsmodels = sm.OLS(y_train, x_train_com_constante, hasconst = True).fit()
modelo_statsmodels.summary()

#Criando um novo conjunto de variáveis explicativas (X)
x = dados[['log_Area', 'log_Dist_Praia']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2811)

x_train_com_constante = sm.add_constant(x_train)
modelo_statsmodels = sm.OLS(y_train, x_train_com_constante, hasconst = True).fit()

modelo_statsmodels.summary()

modelo = LinearRegression()
modelo.fit(x_train, y_train)

#Obtendo o coeficiente de determinação (R²)
print('R² = {}'.format(modelo.score(x_train, y_train).round(3)))

y_previsto = modelo.predict(x_test)

print('R² = %s' % metrics.r2_score(y_test, y_previsto).round(3))