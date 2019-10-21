import numpy as np #矩阵操作
import  pandas as pd #SQL数据处理

from sklearn.metrics import r2_score #评价回归预测模型的性能
import matplotlib.pyplot as plt  #画图
import seaborn as sns

##图形出现在notebook里面
#%matplotlib inline

df = pd.read_csv('FE_boston_housing.csv') #读取上一次运行得到的文件
df.head() #读取前五行

#从原始数据中分离出输入特征x和输出特征y
y = df["MEDV"]
X = df.drop(["MEDV","log_MEDV"],axis = 1)

#特征名称，用于后续显示权重系数对应的特征
feat_names = X.columns
#将数据分割训练数据与测试数据
from sklearn.model_selection import train_test_split

#随机采样20%的数据构建测试样本，作为训练样本
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33,test_size=0.2)
X_train.shape
'''
确定模型类型
3.1尝试缺省参数的线性回归
'''
#线性回归
from sklearn.linear_model import LinearRegression

#1.使用默认配置初始化学习器实例
lr = LinearRegression()

#2.用训练数据训练模型参数
lr.fit(X_train,y_train)

#3.用训练好的模型对测试集进行预测
y_test_pred_lr = lr.predict(X_test)
y_train_pred_lr = lr.predict(X_train)

#查看各特征的权重系数，系数的绝对值大小可看作该特征的重要性
fs = pd.DataFrame({"colums": list(feat_names), "coef":list((lr.coef_.T))})
fs.sort_values(by=['coef'],ascending=False)

#模型评价
#使用r2_score评价模型在测试集和训练集上的性能，并输出评估结果
#测试集
print('The r2 score of LinearRegression on test is',r2_score(y_test,y_test_pred_lr))
#训练集
print('The r2 score of LinerRegression on train is',r2_score(y_train,y_train_pred_lr))
#在训练集上预测残差分布，看是否符合模型假设：噪声为0的高斯噪声
f,ax = plt.subplots(figsize=(7,5))
f.tight_layout()
ax.hist(y_train - y_train_pred_lr,bins=40,label = 'Residuals Linear',color = "b",alpha =0.5) #这里的.5写法卡了半天.......
ax.set_title('Histogram of Residuals')
ax.legend(loc = "best")
plt.show()
#还可以观察预测值与真值的散点图
plt.figure(figsize=(4,3))
plt.scatter(y_train,y_train_pred_lr)
plt.plot([-3,3],[-3,3],'--k')  #数据已经标准化
plt.axis('tight')
plt.xlabel('True price')
plt.ylabel('Predicted price')
plt.tight_layout()
plt.show()
#正则化的线性回归(L2正则_——>岭回归)
from sklearn.linear_model import RidgeCV
#1.设置超参数(正则参数)范围
alphas = [0.01,0.1,1,10,100]
#n_alphas = 20
#alphas = np.logspace(-5,2,n_alphas)

#2.生成一个RidgeCV实例
ridge = RidgeCV(alphas=alphas,store_cv_values=True)

#3.训练模型
ridge.fit(X_train,y_train)

#4.预测
y_test_pred_ridge = ridge.predict(X_test)
y_train_pred_ridge = ridge.predict(X_train)

# 评估，使用r2_score评价模型在测试集和训练集上的性能
print('THe r2 score of RidgeCV on test is',r2_score(y_test, y_test_pred_ridge))
print('THe r2 score of RidgeCV on train is',r2_score(y_train, y_train_pred_ridge))

#可视化
mse_mean = np.mean(ridge.cv_values_,axis=0)
plt.plot(np.log10(alphas),mse_mean.reshape(len(alphas),1))
#这是为了标出最佳参数的位置，不是必须
#plt.plot(np.log10(ridge.alpha_)*np.ones(3), [0.28, 0.29, 0.30])

plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print ('alpha is:', ridge.alpha_)

# 看看各特征的权重系数，系数的绝对值大小可视为该特征的重要性
fs = pd.DataFrame({"columns":list(feat_names), "coef_lr":list((lr.coef_.T)), "coef_ridge":list((ridge.coef_.T))})
fs.sort_values(by=['coef_lr'],ascending=False)
print(fs)

# L1正则-->Lasso
###
#####
#######
from sklearn.linear_model import LassoCV


#1.设置超参数搜索范围
#alphas = [0.01,0.1,1,10,100]

'''
1. 设置超参数搜索范围
Lasso可以自动确定最大的alpha，所以另一种设置alpha的方式是设置最小的alpha值（eps） 和 超参数的数目（n_alphas），
然后LassoCV对最小值和最大值之间在log域上均匀取值n_alphas个
np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),num=n_alphas)[::-1]
'''
lasso = LassoCV()

#3.训练(内含CV)
lasso.fit(X_train,y_train)

#4.测试
y_test_pred_lasso = lasso.predict(X_test)
y_train_pred_lasso = lasso.predict(X_train)

#评估，使用r2_score评价模型在测试集和训练集上的性能
print('The r2 score of LassoCV on test is', r2_score(y_test, y_test_pred_lasso))
print('The r2 score of LassoCV on train is', r2_score(y_train, y_train_pred_lasso))

mses = np.mean(lasso.mse_path_, axis=1)
plt.plot(np.log10(lasso.alphas_), mses)
# plt.plot(np.log10(lasso.alphas_)*np.ones(3), [0.3, 0.4, 1.0])
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print('alpha is:', lasso.alpha_)

# 看看各特征的权重系数，系数的绝对值大小可视为该特征的重要性
fs = pd.DataFrame({"columns":list(feat_names), "coef_lr":list((lr.coef_.T)), "coef_ridge":list((ridge.coef_.T)), "coef_lasso":list((lasso.coef_.T))})
fs.sort_values(by=['coef_lr'],ascending=False)
