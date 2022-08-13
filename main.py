
import pandas as pd #программная библиотека на языке Python для обработки и анализа данных.
import numpy as np #программная библиотека на языке Python для работы с многомерными массивам.
import  matplotlib.pyplot as plt # библиотека на языке программирования Python для визуализации данных двумерной графикой.
import seaborn as sns #библиотека на matplotlib. Предоставляет высокоуровневый интерфейс для рисования привлекательных и информативных статистических графиков.
import scipy.stats as sps #библиотека для языка программирования Python с открытым исходным кодом, предназначенная для выполнения научных и инженерных расчётов.

data=pd.read_csv('data.csv')
print(data)
q=data['quality']
q=set(q)
# Получим основную информацию по данному набору данных.
print(data.describe().T)
"""
data.describe() отражает основную информация по набору, где count - количество ячеек в столбце, mean- среднее значение по столбцу, 
std - стандартное отклонение, min - минимальное значение в столбце, 25%, 50%, 75% - процентный показатель в столбце, 
например для набора 1 2 3 и 25% 50% 75% получим результат 1.5 2 2.5 соответственно, max - максимальное значение в столбце.
"""
print(data.isnull().sum()) # отражает количество пустых ячеек, а так же ячеек, содержащих NaN и null - значения
print()
print(data.dtypes) #это возвращает серию с типом данных каждого столбца.
data1=data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']]
data1
"""
Таким образом, данный набор можно считать идеальным для анализа. Набор представлят из себя числовые значения и не имеет null, NaN и пустых ячеек.
Преобразуем исходные данные для бинарной классификации. Пусть если качество вина меньше 5, то получаем, что вино плохое(0), в противном случае хорошее(1).
"""
data1=data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']]
data1
k1=0
k2=0
for i in range(len(data1)):
  temp=data1.at[i, 'quality']
  if temp>5:
    data1.at[i, 'quality']=1
    k1+=1
  else:
    data1.at[i, 'quality']=0
    k2+=1
# Постром соответствующие наборы для обучения модели.
X1=data1.drop('quality', axis=1)
Y1=data1[['quality']]
y1=data1['quality']
sns.histplot(data=Y1,discrete=1,
                 color='blue')
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

print(k1/(k1+k2),k2/(k1+k2))

print("Тепловая карта корреляции бинарного набора quality:")
sns.heatmap(data1.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=3, linecolor='black')
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

print("Тепловая карта корреляции исходного набора:")
sns.heatmap(data.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=3, linecolor='black')
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


def plotScatterMatrix(df, plotSize, textSize):
  df = df.select_dtypes(include=[np.number])  # keep only numerical columns
  # Remove rows and columns that would lead to df being singular
  df = df.dropna('columns')
  df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
  columnNames = list(df)
  if len(columnNames) > 10:  # reduce the number of columns for matrix inversion of kernel density plots
    columnNames = columnNames[:10]
  df = df[columnNames]
  ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
  corrs = df.corr().values
  # for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
  #     ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
  # plt.suptitle('Scatter and Density Plot')
  plt.rc('font', size=9)
  plt.show()

  import plotly.express as px
  fig = px.box(data, x='quality', y='citric acid')
  fig.show(render='colab')

  def draw_multivarient_plot(data, rows, cols, plot_type):

    column_names = data.columns.values
    number_of_column = len(column_names)
    import matplotlib
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)
    # Set the default text font size
    plt.rc('font', size=20)
    fig, axarr = plt.subplots(rows, cols, figsize=(22, 16))
    counter = 0
    for i in range(rows):
      for j in range(cols):
        if 'violin' in plot_type:
          sns.violinplot(x='quality', y=column_names[counter], data=data, ax=axarr[i][j])
        elif 'box' in plot_type:
          sns.boxplot(x='quality', y=column_names[counter], data=data, ax=axarr[i][j])
        elif 'point' in plot_type:
          sns.pointplot(x='quality', y=column_names[counter], data=data, ax=axarr[i][j])
        elif 'bar' in plot_type:
          sns.barplot(x='quality', y=column_names[counter], data=data, ax=axarr[i][j])

        counter += 1
        if counter == (number_of_column - 1,):
          break

features_list = list(data.columns)
plt.rc('font', size=8)
data[features_list].hist(bins=40, edgecolor='b', linewidth=1.0,
                          xlabelsize=8, ylabelsize=8, grid=True,
                          figsize=(16,9), color='red')

print("Корреляция каждой характеристики относительно quality(бинарный)")
corr_matrix = data1.corr(method='spearman')
corr_matrix['quality'][:-1].plot.bar(color='blue', figsize=(20,10))
plt.ylabel("Correlation with quality")
plt.show()

print("Корреляция каждой характеристики относительно quality(исходный набор)")
corr_matrix = data.corr(method='spearman')
corr_matrix['quality'][:-1].plot.bar(color='blue', figsize=(20,10))
plt.ylabel("Correlation with quality")
plt.show()

from scipy.stats import chi2_contingency
Ind=pd.DataFrame()
Ind.insert(0, 'independence with quality', '')
Ind.insert(0, 'feature', '')
i=0
names = ['alcohol','sulphates','citric acid','fixed acidity','residual sugar','pH','free sulfur dioxide','density','chlorides','total sulfur dioxide','volatile acidity']
for name in names:
  d1=data[[name,'quality']]
  obs = np.array(d1)
  chi2, p, dof, ex = chi2_contingency(obs, correction=False)
  Ind.at[i, 'feature'] =name
  if(p==1):
    Ind.at[i, 'independence with quality'] = 1
  else:
    Ind.at[i, 'independence with quality'] = 0
    print(chi2, p)
  i+=1
print(Ind)

"""
Таким образом, данный набор можно считать идеальным для анализа. Набор представлят из себя числовые значения и не имеет null, NaN и пустых ячеек.
Преобразуем исходные данные для бинарной классификации. Пусть если качество вина меньше 5, то получаем, что вино плохое(0), в противном случае хорошее(1).
"""
data3=data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']]
data3
k1=0
k2=0
for i in range(len(data3)):
  temp=data3.at[i, 'quality']
  if temp>5:
    data3.at[i, 'quality']=2
    k1+=1
  else:
    data3.at[i, 'quality']=1
    k2+=1

from scipy.stats import chi2_contingency
Ind=pd.DataFrame()
Ind.insert(0, 'independence with quality', '')
Ind.insert(0, 'feature', '')
i=0
names = ['alcohol','sulphates','citric acid','fixed acidity','residual sugar','pH','free sulfur dioxide','density','chlorides','total sulfur dioxide','volatile acidity']
for name in names:
  d1=data3[[name,'quality']]
  obs = np.array(d1)
  chi2, p, dof, ex = chi2_contingency(obs, correction=False)
  Ind.at[i, 'feature'] =name
  if(p==1):
    Ind.at[i, 'independence with quality'] = 1
  else:
    Ind.at[i, 'independence with quality'] = 0
    print(chi2, p)
  i+=1
print(Ind)

data1.rename(columns = {'quality' : 'quality_binary'}, inplace = True)
sorted1=data1[['alcohol','sulphates','citric acid','fixed acidity','residual sugar','pH','free sulfur dioxide','density','chlorides','total sulfur dioxide','volatile acidity','quality_binary']]
sorted=data[['alcohol','sulphates','citric acid','fixed acidity','residual sugar','pH','free sulfur dioxide','density','chlorides','total sulfur dioxide','volatile acidity','quality']]
corr_matrix1 = sorted1.corr(method='spearman')
corr_matrix = sorted.corr(method='spearman')
print(corr_matrix)
corr_matrix['quality'][:-1].plot.bar(color='red', figsize=(20,10), legend=True)
corr_matrix1['quality_binary'][:-1].plot.bar(color='blue', figsize=(20,10), legend=True)
plt.ylabel("Correlation with quality")
plt.show()

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=1/3)

import sklearn.externals as extjoblib
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib

"""
Так как наш набор состоит из дискретных значений и основной задачей является 
классификация вина к определенному классу(quality), то в данном случае будет 
некорректным использовать регрессионные модели, так как нам не требуется искать
функцию отображения для отображения входной переменной (x) в непрерывную 
выходную переменную (y).

Мы можем применять нелинейные регрессионные модели для дискретного набора, 
получая при этом статистически достоверные данные (возможно найти нелинейную
функцию Y=f(X), которая будет относительно точно описывать наш дискретный набор),
однако это будет не совсем корректно. Если у данной функции существует 
аналогичный метод, основанный на классификации, то он будет показывать более 
точные результаты, чем регрессионный аналог. Линейная регрессия соответственно
тоже может быть применима для дискретного набора, если все значения дискретного
набора можно интерполировать при помощи прямой Y=kX+b, в нашем примере все 
линейные алгоритмы должны давать статистически недостоверные данные.

Ниже будут рассмотрены все основные методы, в том числе и регрессивные, чтобы
проиллюстрировать приведенные выше замечания.
"""
from sklearn.linear_model import Ridge # Метод Тихонова (линейная регрессия)
from sklearn.linear_model import Lasso # Лассо (линейная регрессия)
from sklearn.linear_model import ElasticNet # Эластичная сеть (линейная регрессия)
from sklearn.linear_model import LinearRegression # Линейная рергессия
from sklearn.linear_model import Lars #Метод наименьших углов (линейная регрессия)
from sklearn.linear_model import BayesianRidge #Байесовская гребневая регрессия (линейная регрессия)
from sklearn.svm import LinearSVC, SVC #Метод опорных векторов(регрессия)
from sklearn.neighbors import  KNeighborsClassifier # K-соседи
from sklearn.neighbors import  KNeighborsRegressor # K-соседи (регрессия)
from sklearn.tree import  DecisionTreeClassifier as DTC#Дерево решений
from sklearn.tree import  DecisionTreeRegressor as DTR # Деревья регрессии (регрессия)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as RFR #Рандомный лес (регрессия)
from sklearn.ensemble import AdaBoostClassifier as ABC#Классификатор AdaBoost
from sklearn.ensemble import AdaBoostRegressor as ABR #Регрессор AdaBoost
from sklearn.ensemble import BaggingRegressor as BR#Bagging (регрессия)
from sklearn.ensemble import BaggingClassifier as BC#Bagging
from sklearn.ensemble import ExtraTreesRegressor #Экстра-деревья (регрессия)
from sklearn.ensemble import ExtraTreesClassifier  #Экстра-деревья
from sklearn.ensemble import GradientBoostingRegressor as GBR #Градиентный boosting (регрессия)
from sklearn.ensemble import GradientBoostingClassifier as GBC #Градиентный boosting
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
"""
Приведем данные к нормальному виду, то есть приведем к дисперсии 1 и матожиданию 0
"""
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

models = {

  'KNeighborsClassifier': KNeighborsClassifier(),
  'DecisionTreeClassifier': DTC(random_state=42, max_depth=8),
  'RandomForestClassifier': RandomForestClassifier(n_estimators=90, max_depth=8),
  'AdaBoostClassifier': ABC(n_estimators=90),
  'BaggingClassifier': BC(n_estimators=5),
  'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=90, max_depth=8),
  'GradientBoostingClassifier': GBC(n_estimators=10)
}

scores = {}
for model_name, model in models.items():
  model.fit(X_train_std, y_train)
  scores[(model_name, 'Method')] = model_name
  scores[(model_name, 'train')] = model.score(X_train_std, y_train)
  scores[(model_name, 'test')] = model.score(X_test_std, y_test)
res = pd.Series(scores).unstack().sort_values(['test'], ascending=False)
res = res.reset_index()
del res['index']
print(res)

# Постром соответствующие наборы для обучения модели, не прибегая к бинарной классификации.
X=data.drop('quality', axis=1)
Y=data[['quality']]
y=data['quality']

"""
Рассмотрим, насколько сильно неупорядоченны наши даннные. Для этого построим диаграмму распределения величины quality, а так же 
тепловую карту корреляции. Корреляция отражает статистическую взаимосвязь двух или более случайных величин. Таким образом, получим, 
как каждая из фич зависит от другой подобной ей.
"""

sns.histplot(data=Y,
                 kde=True, fill=True, discrete=1,
                 color='blue', line_kws={'lw':5,'ls':'-','color':'red'})
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
print("Тест Андерсона-Дарлинга:")
from scipy.stats import anderson
stat, cv,sv = anderson(y, dist='norm')
print(stat,cv,sv)

"""
Таким образом, получаем, что данные распределились нормальным образом.
"""
print("Тепловая карта корреляции:")
sns.heatmap(data.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=3, linecolor='black')
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

print("Корреляция каждой характеристики относительно quality")
corr_matrix = data.corr(method='spearman')
corr_matrix['quality'][:-1].plot.bar(color='blue', figsize=(20,10))
plt.ylabel("Correlation with quality")
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

models = {

  'KNeighborsClassifier': KNeighborsClassifier(),
  'DecisionTreeClassifier': DTC(random_state=42, max_depth=8),
  'RandomForestClassifier': RandomForestClassifier(n_estimators=90, max_depth=6),
  'AdaBoostClassifier': ABC(n_estimators=90),
  'BaggingClassifier': BC(n_estimators=5),
  'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=90, max_depth=8),
  'GradientBoostingClassifier': GBC(n_estimators=10)
}

scores = {}
for model_name, model in models.items():
  model.fit(X_train_std, y_train)
  scores[(model_name, 'Method')] = model_name
  scores[(model_name, 'train')] = model.score(X_train_std, y_train)
  scores[(model_name, 'test')] = model.score(X_test_std, y_test)

res = pd.Series(scores).unstack().sort_values(['test'], ascending=False)
res = res.reset_index()
del res['index']
print(res)

res2=pd.DataFrame()
res2.insert(0, 'TrainRegressor', '')
res2.insert(0, 'TestRegressor', '')
res2.insert(0, 'TrainClassifier', '')
res2.insert(0, 'TestClassifier', '')
res2.insert(0, 'Method', '')
k=0
for i in range(len(res)):
  temp=res.at[i, 'Method']
  if temp.find('Classifier')!=-1:
    name=temp.replace("Classifier", "")
    res2.at[k, 'Method'] = name
    res2.at[k, 'TrainClassifier'] = res.at[i, 'train']
    res2.at[k, 'TestClassifier'] = res.at[i, 'test']
    k+=1
print(res2)


import numpy as np
import matplotlib.pyplot as plt
"""
Попробуем преобразовать данные для получения более достоверных результатов при 
помощи метода k-средних — наиболее популярного метод кластеризации.
"""
count_data=data.groupby('quality')['quality'].count()
print(count_data)
count_data.plot(kind='bar')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
"""
Попробуем преобразовать данные для получения более достоверных результатов при 
помощи метода k-средних — наиболее популярного метод кластеризации.
"""
count_data=data.groupby('quality')['quality'].count()
print(count_data)
count_data.plot(kind='bar')
plt.show()

from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300, n_init=12, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
print(wcss)
f3, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(1,11),wcss,marker='*')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

resTr=pd.DataFrame()
resTr.insert(0, 'Method', '')
resTe=pd.DataFrame()
resTe.insert(0, 'Method', '')

k=50
kmeans=KMeans(init='random',n_clusters=k,random_state=42).fit(X)

labels=pd.Series(kmeans.labels_,name='c luster_number')
print(labels.value_counts(sort=False))

ax=labels.value_counts(sort=False).plot(kind='bar')
ax.set_xlabel('cluster_number')
ax.set_ylabel('count')
y=labels

from sklearn.model_selection import train_test_split
y=labels
X=data.drop('quality', axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
models = {
    'KNeighborsClassifier':  KNeighborsClassifier(),
    'DecisionTreeClassifier': DTC(random_state=42,max_depth=7),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=90,max_depth=7),
    'AdaBoostClassifier': ABC( n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=5),
    'BaggingClassifier':BC(n_estimators=5),
    'ExtraTreesClassifier':ExtraTreesClassifier(n_estimators=90,max_depth=7),
    'GradientBoostingClassifier': GBC( n_estimators=10)
}

i=-1
for model_name, model in models.items():
    i+=1
    model.fit(X_train_std, y_train)
    resTr.at[i, 'Method'] = model_name
    resTe.at[i, 'Method'] = model_name
    resTe.at[i, 'test'+str(k)] = round(model.score(X_test_std, y_test),2)
    resTr.at[i, 'train'+str(k)] = round(model.score(X_train_std, y_train),2)
    if(i==6):
      i=-1

print(resTe)
print(resTr)

from sklearn.model_selection import train_test_split
y=labels
X=data.drop('quality', axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

models = {
    'KNeighborsClassifier':  KNeighborsClassifier(),
    'DecisionTreeClassifier': DTC(random_state=42,max_depth=5),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=90,max_depth=6),
    'AdaBoostClassifier': ABC( n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=5),
    'BaggingClassifier':BC(n_estimators=5),
    'ExtraTreesClassifier':ExtraTreesClassifier(n_estimators=90,max_depth=9),
    'GradientBoostingClassifier': GBC( n_estimators=10)
}

i=-1
for model_name, model in models.items():
    i+=1
    model.fit(X_train_std, y_train)
    resTr.at[i, 'Method'] = model_name
    resTe.at[i, 'Method'] = model_name
    resTe.at[i, 'test'+str(k)] = round(model.score(X_test_std, y_test),2)
    resTr.at[i, 'train'+str(k)] = round(model.score(X_train_std, y_train),2)
    if(i==6):
      i=-1

RFC1=RandomForestClassifier
clf=RFC1(n_estimators=100,max_depth=9,max_features=1,min_samples_split=2)
clf.fit(X_train_std, y_train)
Tr=clf.score(X_train_std, y_train)
Te=clf.score(X_test_std, y_test)
print(Tr,Te)

res3=pd.DataFrame()
res3.insert(0, 'TrainRegressor', '')
res3.insert(0, 'TestRegressor', '')
res3.insert(0, 'TrainClassifier', '')
res3.insert(0, 'TestClassifier', '')
res3.insert(0, 'Method', '')
k=0
for i in range(len(res)):
  temp=res.at[i, 'Method']
  if temp.find('Classifier')!=-1:
    name=temp.replace("Classifier", "")
    res3.at[k, 'Method'] = name
    res3.at[k, 'TrainClassifier'] = res.at[i, 'train']
    res3.at[k, 'TestClassifier'] = res.at[i, 'test']
    k+=1
print(res3)

