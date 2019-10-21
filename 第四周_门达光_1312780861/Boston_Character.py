#波士顿房价预测案例——特征工程
import numpy as np #矩阵操作
import pandas as pd #SQL数据处理

#path to where the data lies
#dpath = './data/'
df = pd.read_csv(r'/home/m/Downloads/boston_housing.csv')

print(df.head())#观察前五行看看读到的数据大概是什么样子呢(可有可无)
df.info() #显示数据的基本信息,样本数目(行数)/特征维数(列数)/空样本数目/数据类型
#特征工程
#3.1数据去噪
df = df[df.MEDV < 50]
print(df.shape)
#3.2数据分离
#从原始数据分离输入特征X 和 输出y
y = df["MEDV"]
X = df.drop("MEDV",axis=1)
#尝试对y(房屋价格)做log变换，对log变换后的价格进行估计
log_y = np.log1p(y)
#3.3离散型特征编码
# RAD的含义是距离高速公路的便利指数。虽然给的是数值，但实际是索引，可以换成离散特征/类别特征编码试试
X["RAD"].astype("object")
X_cat = X["RAD"]
X_cat = pd.get_dummies(X_cat,prefix="RAD")

X = X.drop("RAD",axis = 1)
#特征名称，用于保存特征工程结果
feat_names = X.columns
print(X_cat.head())
'''
3.4数值特征预处理
3.4.1数值特征标准化
'''
#数据标准化
from sklearn.preprocessing import StandardScaler
#分别初始化对特征和目标值的标准化器.
ss_X = StandardScaler()
ss_y = StandardScaler()

ss_log_y = StandardScaler()
#分别对训练和测试数据的特征及目标值进行标准化处理
#对训练数据，先调用fit方法训练模型，得到模型参数；然后对训练数据进行transform
X = ss_X.fit_transform(X)
#对Y 作标准化不是必须的
#对Y 作标准花的好处是不同问题的w差异不大，同时正则参数范围也有限
y = ss_y.fit_transform(y.values.reshape(-1,1))
log_y = ss_y.fit_transform(log_y.values.reshape(-1,1))

# 4.保存特征工程的结果到文件，供机器学习模型使用
fe_data = pd.DataFrame(data=X,columns = feat_names,index = df.index)
fe_data = pd.concat([fe_data,X_cat],axis = 1,ignore_index=False)
#加上标签
fe_data["MEDV"] = y
fe_data["log_MEDV"] = log_y

#保存结果到文件
fe_data.to_csv("FE_boston_housing.csv",index=False)


