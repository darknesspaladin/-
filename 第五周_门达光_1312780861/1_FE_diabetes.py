#数据探索/预处理
import numpy as np
import pandas as pd

train = pd.read_csv('pima-indians-diabetes.csv')
train.head()
#观察数据集中有无缺失，例如血压和体重指数如果为零，显然不对。
#利用上一例子中统计的方法实现统计0项,多了isnull方法
NaN_col_names = ['Plasma_glucose_concentration','blood_pressure','Triceps_skin_fold_thickness','serum_insulin','BMI']
train[NaN_col_names] = train[NaN_col_names].replace(0,np.NaN)
print(train.isnull().sum())
#对缺失值较多的的特征，新增一个特征，表明是还是不是缺失值
train['Triceps_skin_fold_thickness_Missing'] = train['Triceps_skin_fold_thickness'].apply(lambda x:1 if pd.isnull(x) else 0)
train[['Triceps_skin_fold_thickness','Triceps_skin_fold_thickness_Missing']].head()

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x = "Triceps_skin_fold_thickness_Missing",hue = 'Target',data = train)
plt.show()

#由于缺失值太多，干脆开一个新的字段，表明是缺失值还是不是缺失值
train['serum_insulin_Missing'] = train['serum_insulin'].apply(lambda x: 1 if pd.isnull(x) else 0)
#由于缺失值和目标没有关系，直接用中值填补0项
medians = train.median()
train = train.fillna(medians)
print(train.isnull().sum())
#数据标准化
#get labels
y_train = train['Target']
X_train = train.drop(['Target'],axis = 1)
#用于保存特征工程之后的结果
feat_names = X_train.columns
from sklearn.preprocessing import StandardScaler
#初始化特征标准器
ss_X = StandardScaler()
#分别对训练和测试数据进行处理
X_train = ss_X.fit_transform(X_train)

#特征处理结果存为文件。csv格式
X_train = pd.DataFrame(columns = feat_names, data=X_train)
train = pd.concat([X_train,y_train],axis = 1)
train.to_csv('FE_pima-indians-diabetes.csv',index=False,header=True)
train.head()




