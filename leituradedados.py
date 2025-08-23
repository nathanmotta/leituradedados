#r2 com dados reais (plano cartesiano e gráfico de dispersão de dados)

import matplotlib.pyplot as plt #biblioteca para a visualização de dados em Python
idades = [70, 65, 72, 63, 71, 64, 60, 64, 67]
alturas = [175, 170, 198, 174, 155, 130, 105, 145, 162]
pessoas = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
plt.scatter(idades, alturas) #cria um gráfico de dispersão usando os dados anteriores
for pessoas_count, idades_count, alturas_count in zip(pessoas, idades, alturas):
    plt.annotate(pessoas_count, xy=(idades_count, alturas_count), xytext=(5, -5), textcoords='offset points')
plt.title("Idades vs Alturas de amigos")
plt.xlabel("Idades")
plt.ylabel("Alturas")
plt.show()




Gráfico 3D

#espaço r3 com dados reais (gráfico de dispersão 3d)
from mpl_toolkits import mplot3d
idades = [70, 65, 72, 63, 71, 64, 60, 64, 67]
alturas = [175, 170, 198, 174, 155, 130, 105, 145, 162]
peso = [65, 45, 78, 79, 82, 64, 77, 95, 110]
pessoas = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(idades, alturas, peso, 'gray')
ax.set_xlabel('idades')
ax.set_ylabel('alturas')
ax.set_zlabel('peso');



Criando gráficos com Pandas

from random import *
x=[]
for i in range (0, 20):
    z = round(float(uniform(0.0, 10.10)), 2)
    x.append(z)
len(x)

y=[]
for i in range(0,20):
    y.append(int(uniform(0.0,100)))

z = x,y

import pandas as pd
df = pd.DataFrame(z, index=['x', 'y']).T
df




exemplo 2:



from random import *
x=[]
for i in range (0,20):
    z = round(float(uniform(0.0, 10.10)), 2)
    x.append(z)

len(x)

x

y=[]
for i in range(0,20):
    y.append(int(uniform(0.0,100)))

z = x,y

z

#criando DataFrame em Python - Usando Pandas

import pandas as pd
df = pd.DataFrame(z, index=['x','y']).T
df

df['x']

#gerando histograma de valores da coluna x
plt.hist(df.x)

#gerando histograma de valores da coluna y
plt.hist(df.y)









Distribui??o de problemas card?acos:


# Import uniform from random module instead of sympy
from random import uniform
import numpy as np
import pandas as pd

# Define n (which was missing in the original code)
n = 100  # Assuming n=100, adjust as needed

# Generate height data
alt = np.random.normal(1.75, 0.30, n).tolist()

# c?lculo do peso
peso = []
for i in alt:
  a = round(float(uniform(-5, 5)), 2)  # varia??o aleat?ria
  b = 50 * i
  c = a + b
  peso.append(c)

# c?lculo do IMC
IMC = []
for i, j in zip(peso, alt):
  e = i / (j**2)
  e = round(e, 2)
  IMC.append(e)

# gera??o de SEXO
SEXO = []
for i in range(n):
  value = np.random.uniform(0, 1)
  if value > 0.5:
      SEXO.append('M')
  else:
      SEXO.append('F')

# classifica??o card?aca
CARDIACO = []
for i in range(n):
  if IMC[i] > 40:
      CARDIACO.append('S')  # Sim, problema
  elif 30 < IMC[i] <= 40:
      CARDIACO.append('P')  # Propens?o
  else:
      CARDIACO.append('N')  # Normal

# cria??o do dataframe
df = pd.DataFrame({
  'Altura': alt,
  'Peso': peso,
  'IMC': IMC,
  'SEXO': SEXO,
  'CARDIACO': CARDIACO
})

df.to_csv('df.csv', index=False)

# fun??o m?dia
def media(tabela):
  soma = sum(tabela)
  media = soma / len(tabela)
  media = round(media, 2)
  print(media)

print("M?dia IMC:")
media(df.IMC)

print("M?dia Peso:")
media(df.Peso)

print("M?dia Altura:")
media(df.Altura)

# distribui??o de sexo
freq_m = (df.SEXO == 'M').sum()
freq_f = (df.SEXO == 'F').sum()
print("O percentual de homens ?:", round((freq_m/n*100), 2))
print("O percentual de mulheres ?:", round((freq_f/n*100), 2))

# distribui??o card?aca
freq_s = (df.CARDIACO == 'S').sum()
freq_p = (df.CARDIACO == 'P').sum()
freq_n = (df.CARDIACO == 'N').sum()
print("O percentual de pessoas com problemas card?acos ?:", round((freq_s/n*100), 2))
print("O percentual de pessoas com tend?ncias a problemas card?acos ?:", round((freq_p/n*100), 2))
print("O percentual de pessoas sem tend?ncia a problemas card?acos ?:", round((freq_n/n*100), 2))

# gr?fico para visualizar
import matplotlib.pyplot as plt

df['CARDIACO'].value_counts().plot(kind='bar')
plt.title("Distribui??o de Problemas Card?acos")
plt.xlabel("Categoria")
plt.ylabel("Frequ?ncia")
plt.show()





Criando tabelas com Pandas:


# Importando o pandas
import pandas as pd

# Criando um DataFrame simples
df = pd.DataFrame({
    'nome': ['Ana', 'Bruno', 'Carlos', 'Lucas', 'Ricardo', 'Gabriel', ''],
    'idade': [23, 35, 31, 28, 18, 15, 10],
    'salario': [3000, 5000, 4500, 3000, 2500, 5000, 100]
})

df



#Visualiza??o r?pida de dados:

#In?cio
df.head() #mostra os 5 primeiros




#fim
df.tail() #mostra os 5 ?ltimos



#Mostrando informa??es da DataFrame:

df.info()





#Aplicando describe e medidas da Estatistica Descritiva


#resumo geral
df.describe()



#medida espec?fica - m?dia idade
df['idade'].mean()



#medida espec?fica - mediana idade
df['idade'].median()




#medida espec?fica - desvio padr?o idade
df['idade'].std()



#medida espec?fica - min idade
df['idade'].min()



#medida espec?fica - max idade
df['idade'].max()




#Sele??o de linhas e colunas de um dataframe:


#o mais simples
df['idade']



#usando loc
df.loc[1] #segunda linha



df.iloc[0:2] #selecionando um intervalo de 0 a 1 (n?o inclui o 2) de linhas





#Manipulando um DataFrame - B?sico:


#multiplicando a coluna idade por 0.6
df['idade'] * 0.6



#salvando a manipula??o anterior
df['idade'] = df['idade'] * 0.6
df



#Dividindo a coluna idade por 0.6
df['idade'] / 0.6



#salvando a manipula??o anterior
df['idade'] = df['idade'] / 0.6
df


















