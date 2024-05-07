import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import shuffle
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin


os.chdir("D:/projectST/data/final/WHO_grades/TRA/")
inputfile = "6+14.txt"

# 加载数据
labels = ['positive', 'negative']
tick_marks = np.array(range(len(labels))) + 0.5


def readData():
    data = pd.read_table(inputfile, sep='\t', encoding='utf-8')
    data = data.values.astype(np.float32)
    shuffle(data)  # 随机打乱
    data_train = data[:int(1 * len(data)), :]  # 数据集矩阵
    return data_train


def train(data_train):
    x = data_train[:, 1:20] * 30  # 放大特征，矩阵2维
    y = data_train[:, 0].astype(int)  # 矩阵1维
    return x, y


if __name__ == '__main__':
    data_train = readData()
    x, y = train(data_train)


# 定义弱学习器
model = MLPClassifier(
    hidden_layer_sizes=(100,100,50,25),
    random_state = 420,
    max_iter = 5000,
   learning_rate_init = 0.001,
    momentum = 0.9,
    batch_size = 8,
    solver = "adam",
    activation = "relu"
) # 使用两层，每层50个节点


# 自定义分类器
class CustomLGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, lgb_params=None, base_weight=0.5, lgb_weight=0.5):
        self.base_estimator = base_estimator
        self.lgb_params = lgb_params or {}
        self.base_weight = base_weight
        self.lgb_weight = lgb_weight

    def fit(self, X, y):
        # 使用基分类器进行训练
        self.base_estimator.fit(X, y)

        # 使用基分类器的预测概率作为特征来训练LightGBM模型
        base_train_proba = self.base_estimator.predict_proba(X)
        X_lgb = np.hstack((X, base_train_proba))
        lgb_train_data = lgb.Dataset(X_lgb, label=y)

        self.lgbm = lgb.train(self.lgb_params, lgb_train_data)
        return self

    def predict(self, X):
        base_preds = self.base_estimator.predict(X)
        base_proba = self.base_estimator.predict_proba(X)
        X_lgb = np.hstack((X, base_proba))
        lgb_preds = np.round(self.lgbm.predict(X_lgb))
        final_preds = np.round((self.base_weight * base_preds + self.lgb_weight * lgb_preds) / (self.base_weight + self.lgb_weight))
        return final_preds

    def predict_proba(self, X):
        base_proba = self.base_estimator.predict_proba(X)
        X_lgb = np.hstack((X, base_proba))
        lgb_proba = self.lgbm.predict(X_lgb)
        final_proba = (self.base_weight * base_proba[:, 1] + self.lgb_weight * lgb_proba) / (self.base_weight + self.lgb_weight)
        return np.vstack((1 - final_proba, final_proba)).T


# 定义 Boosting 集成模型

model_ensemble = CustomLGBMClassifier(base_estimator=model,lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': 100,
    })

smo = SMOTE(random_state=42)
cv = StratifiedKFold(n_splits=5)  # 导入该模型，后面将数据划分5份

mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
mean_fpr = np.linspace(0, 1, 100)
cnt = 0
for i, (train, test) in enumerate(cv.split(x, y)):  # 利用模型划分数据集和目标变量 为一一对应的下标
    cnt += 1

    x_train, y_train = smo.fit_resample(x[train], y[train])
    print(Counter(y_train))
    probas_ = model_ensemble.fit(x_train, y_train).predict_proba(x[test])  # 训练模型后预测每条样本得到两种结果的概率
    #print(probas_)
    y_pred = model_ensemble.predict(x[test])
    #print(y_pred)
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
    mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
    mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点
    roc_auc = auc(fpr, tpr)  # 求auc面积


    plt.plot(fpr, tpr,  label='AUC_one fold = {0:.4f}'.format(roc_auc), lw=1.5)  # 画出当前分割数据的ROC曲线


    accuracy = accuracy_score(y[test], y_pred)
    print("accuracy: %.2f%%" % (accuracy * 100.0))

    recall = recall_score(y[test], y_pred)
    print("recall: %.2f%%" % (recall * 100.0))

    prec = precision_score(y[test], y_pred)
    print("prec: %.2f%%" % (prec * 100.0))

    f1 = f1_score(y[test], y_pred)
    print("f1: %.2f%%" % (f1 * 100.0))


plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))  # 画对角线

mean_tpr /= cnt  # 求数组的平均值
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'b--', label='AUC_average = {0:.4f}'.format(mean_auc), lw=4)

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
#plt.title('model')
plt.legend(loc="lower right")
plt.show()















