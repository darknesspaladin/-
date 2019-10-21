import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('pima-indians-diabetes.csv') #读取数据文件
c = train.head(8)
print(c)
print(train.shape)
train.info()
print(train.describe()) #查看数值型特征的基本统计量

#创建数值为0时无意义的指标集合
NaN_col_names = ['Plasma_glucose_concentration','blood_pressure','Triceps_skin_fold_thickness','serum_insulin','BMI']
print((train[NaN_col_names] == 0).sum()) #统计特征中0的数量。

#查看每个变量的分布，及其与标签之间的关系
sns.countplot(train['Target'])
plt.xlabel('Diabetes')
plt.ylabel('Number of occurrence')
plt.show()
#怀孕次数
fig = plt.figure()
sns.countplot(train['pregnants'])
plt.xlabel('number of pregnants')
plt.ylabel('number of occurrences')
plt.show()


#将怀孕次数与发病次数整合到同一个图表
sns.countplot(x='pregnants',hue='Target',data = train)

plt.show()
#怀孕次数和是否得病好像还真有关系！！！
#空腹血浆葡萄糖浓度与糖尿病的关系
fig = plt.figure()
img1 = sns.distplot(train.Plasma_glucose_concentration,kde = False)
plt.xlabel('Plasma_glucose_concentration')
plt.ylabel('Number of occurrences')

img2 = sns.violinplot(x='Target',y = "Plasma_glucose_concentration",data = train,hue = 'Target')
plt.xlabel('Diabetes',fontsize=12)
plt.ylabel('Plasma_glucose_concentration',fontsize=12)
plt.show()
#BMI 体重指数
BMIDF = train.groupby(['BMI','Target'])['BMI'].count().unstack('Target').fillna(0)
BMIDF[[0,1]].plot(kind = 'bar',stacked = True)
plt.show()
#糖尿病家系作用 Diabetes_pedigree_function
fig = plt.figure()
sns.distplot(train.Diabetes_pedigree_function,kde = False)
plt.xlabel('Diabetes_pedigree_function')
plt.ylabel('frequency')

DF = train.groupby(['Diabetes_pedigree_function','Target'])['Diabetes_pedigree_function'].count().unstack('Target').fillna(0)
DF[[0,1]].plot(kind='bar',stacked = True)
plt.show()
#年龄分布与target之间的关系
fig = plt.figure()
sns.distplot(train.Age,kde = False)
plt.xlabel('Age')
plt.ylabel("Frequency")

#特征之间的关系
data_corr = train.corr().abs()

plt.subplots(figsize=(13,9))
sns.heatmap(data_corr,annot=True)
for feature in train.columns:
    sns.distplot(train[feature],kde = False)
    plt.show()
