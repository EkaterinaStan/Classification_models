#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
# загружаем набор данных по цветкам ириса, в нем три класса ириса, по 50 образцов каждый
# каждый образец описан 4 характеристиками: длина чашелистика, ширина чашелистика, длина лепестка, ширина лепестка
iris = load_iris()
X = np.array(iris.data)
y = np.array(iris.target)
# разбиваем датасет на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# строим простую логистическую регрессию
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(f'Коэффициенты логистичекой регресии {log_reg.coef_}')
y_predict_log_reg = log_reg.predict(X_test)

# оцениваем сингулярные значения независимых компонент, чтобы понять сколько их взять в анализ
n_columns = len(X[0])
pca = PCA(n_components = n_columns)
pca.fit(X_train)
plt.bar(range(1, n_columns + 1), pca.singular_values_)
# cжимаем данные до одной независимой компоненты
pca = PCA(1, random_state = 1)
X_train_compressed = pca.fit_transform(X_train)

# строим логистичекую регрессию на основе pca для предсказния класса цветка
log_reg_pca = LogisticRegression()
log_reg_pca.fit(X_train_compressed, y_train)
print(f'Коэффициенты логичтической регресии на основе рса  {log_reg_pca.coef_}')
y_pred_log_reg_pca = log_reg_pca.predict(pca.transform(X_test))

# обучаем наивный байесовский класификатор на тестовой выборке
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# с помощью байесовского классификатора прогнозирум к какому классу относится цветок
y_pred_gnb = gnb.predict(X_test)

#считаем метрику точности для каждого типа моделей
log_reg_error = (accuracy_score(y_test, y_pred_log_reg) * 100).round(2)
log_reg_pca_error = (accuracy_score(y_test, y_pred_log_reg_pca) * 100).round(2)
gnb_error = (accuracy_score(y_test, y_pred_gnb) * 100).round(2)
print(f'точность логистической регресси {log_reg_error}')
print(f'точность логистической регресси + pca {log_reg_pca_error}')
print(f'точность байесовского классификатора {gnb_error }')


# In[ ]:





# In[ ]:





# In[ ]:




