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







# Import uniform from random module instead of sympy
from random import uniform
import numpy as np
import pandas as pd

# Define n (which was missing in the original code)
n = 100  # Assuming n=100, adjust as needed

# Generate height data
alt = np.random.normal(1.75, 0.30, n).tolist()

# cálculo do peso
peso = []
for i in alt:
    a = round(float(uniform(-5, 5)), 2)  # variação aleatória
    b = 50 * i
    c = a + b
    peso.append(c)

# cálculo do IMC
IMC = []
for i, j in zip(peso, alt):
    e = i / (j**2)
    e = round(e, 2)
    IMC.append(e)

# geração de SEXO
SEXO = []
for i in range(n):
    value = np.random.uniform(0, 1)
    if value > 0.5:
        SEXO.append('M')
    else:
        SEXO.append('F')

# classificação cardíaca
CARDIACO = []
for i in range(n):
    if IMC[i] > 40:
        CARDIACO.append('S')  # Sim, problema
    elif 30 < IMC[i] <= 40:
        CARDIACO.append('P')  # Propensão
    else:
        CARDIACO.append('N')  # Normal

# criação do dataframe
df = pd.DataFrame({
    'Altura': alt,
    'Peso': peso,
    'IMC': IMC,
    'SEXO': SEXO,
    'CARDIACO': CARDIACO
})

df.to_csv('df.csv', index=False)

# função média
def media(tabela):
    soma = sum(tabela)
    media = soma / len(tabela)
    media = round(media, 2)
    print(media)

print("Média IMC:")
media(df.IMC)

print("Média Peso:")
media(df.Peso)

print("Média Altura:")
media(df.Altura)

# distribuição de sexo
freq_m = (df.SEXO == 'M').sum()
freq_f = (df.SEXO == 'F').sum()
print("O percentual de homens é:", round((freq_m/n*100), 2))
print("O percentual de mulheres é:", round((freq_f/n*100), 2))

# distribuição cardíaca
freq_s = (df.CARDIACO == 'S').sum()
freq_p = (df.CARDIACO == 'P').sum()
freq_n = (df.CARDIACO == 'N').sum()
print("O percentual de pessoas com problemas cardíacos é:", round((freq_s/n*100), 2))
print("O percentual de pessoas com tendências a problemas cardíacos é:", round((freq_p/n*100), 2))
print("O percentual de pessoas sem tendência a problemas cardíacos é:", round((freq_n/n*100), 2))

# gráfico para visualizar
import matplotlib.pyplot as plt

df['CARDIACO'].value_counts().plot(kind='bar')
plt.title("Distribuição de Problemas Cardíacos")
plt.xlabel("Categoria")
plt.ylabel("Frequência")
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






Numpy

O b?sico de Numpy

#criando um vetor

import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
a


#criando uma matriz
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 11]])
b


np.zeros(10) #? um vetor de zeros


np.ones(10) #? um array de uns


np.arange(4) #intervalo come?ando no zero


np.arange(2, 9, 2) #de 2 at? 9, de 2 em 2


np.linspace(0, 20, num=5) #dados espa?ados linearmente, de 0 a 20 de 5 em 5


np.sort(a) #classificando em ordem crescente


np.sort(b) #classificando em ordem crescente



#juntar dois vetores
z = np.array([1, 2, 3, 4])
w = np.array([4, 6, 7, 8])
np.concatenate((z, w)) #juntando (concatenando)




#Vetores, Norma e Dire??o:



# Definindo vetores
v1 = np.array([3, 4])
v2 = np.array([4, 3])

# Calculando a norma (magnitude)
norm_v1 = np.linalg.norm(v1)
norm_v2 = np.linalg.norm(v2)

print(f"Norma de v1: {norm_v1:.2f}")
print(f"Norma de v2: {norm_v2:.2f}")

# Dire??o = vetor unit?rio
u1 = v1 / norm_v1
u2 = v2 / norm_v2

print(f"Dire??o de v1 (unit?rio): {u1}")
print(f"Dire??o de v2 (unit?rio): {u2}")




#Multiplica??o Escalar e Sentido:



# Multiplicando por um escalar positivo
v3 = 2 * v1
print(f"2 * v1 = {v3}")

# Multiplicando por um escalar negativo (muda o sentido)
v4 = -2 * v1
print(f"-2 * v1 = {v4}")




#Matrizes e Multiplica??o de Matrizes:




# Criando matrizes A (1x2) e B (2x1)
A = np.array([[1, 2]])
B = np.array([[3], [4]])

# Verifica as formas
print("Forma de A:", A.shape)
print("Forma de B:", B.shape)

# Multiplica??o de Matrizes
C = np.dot(A, B)
print("A * B =", C)




#Transposi??o de Matrizes


# Transpondo uma matriz
A_T = A.T
print("Matriz A transposta:\n", A_T)




#Invers?o de Matrizes e Matriz Identidade:


from numpy.linalg import inv, det

# Matriz 2x2
M = np.array([[4, 7], [2, 6]])

# Verificando se ? invert?vel
det_M = det(M)
print("Determinante de M:", det_M)

if det_M != 0:
    M_inv = inv(M)
    print("Matriz inversa de M:\n", M_inv)

    # Verificando a multiplica??o com a inversa = identidade
    identidade = np.dot(M, M_inv)
    print("M * M_inv (deve ser identidade):\n", identidade)
else:
    print("Matriz n?o invert?vel.")













