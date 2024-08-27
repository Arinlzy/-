import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 加载鸢尾花数据集
X, y = load_iris(return_X_y=True)
sc=StandardScaler()
sc.fit(X)
X=sc.transform(X)
# 将数据集划分为训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 调用sklearn库的朴素贝叶斯模型
nb_sklearn = GaussianNB()
# 训练模型
nb_sklearn.fit(X_train, y_train)
# 预测结果
y_pred_sklearn = nb_sklearn.predict(X_test)
# 评估在测试集上的预测准确率
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"调用sklearn库的朴素贝叶斯分类器的准确率: {acc_sklearn:.2f}")


# 构建自定义的朴素贝叶斯模型
class myGaussianNB:
    """
    原理：
        贝叶斯公式: P(Y|X)=P(X|Y)P(Y)/P(X);
        属性条件独立性假设: P(X|Y)=P(c1|Y)P(c2|Y)...P(cn|Y);
        GaussianNB是朴素贝叶斯分类器的一种,它假设数据的每个属性都服从高斯分布。
    -----------------------
    训练:
	对每个类别y：
        提取属于该类别的样本；
        计算该类别样本在每个属性上的均值、方差；
        计算该类别的先验概率 P(Y)。
    -----------------------
    预测：
    对每个类别 y:
        获取该类别的先验概率 P(Y);
        计算每个属性在该类别下的条件概率P(c|Y);
        根据属性独立性假设,将每个属性的概率密度相乘得到 P(X|Y);
        根据贝叶斯公式计算后验概率 P(Y|X) = P(X|Y) * P(Y);
    选择后验概率最大的类别作为预测结果.
    """

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = {}

        for cls in self.classes:
            ## 按类别提取样本
            X_cls = X_train[y_train == cls]
            ## 计算均值
            X_mean = np.mean(X_cls, axis=0)
            ## 计算方差
            X_var = np.var(X_cls, axis=0)
            ## 计算先验P(Y)
            X_prior = len(X_cls) / len(X_train)
            self.params[cls] = {
                'mean':X_mean,
                'var':X_var,
                'prior':X_prior
            }

    def predict(self, X):
        n_samples, n_features = X.shape
        predictions = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                # 获取先验概率
                prior = np.log(self.params[cls]['prior'])
                # 计算每个属性的类条件概率P(C|Y)
                conditional_c=np.log(self.Gaussian_pdf(cls, x))
                # 依据P(C|Y)计算类条件概率P(X|Y)
                conditional = np.sum(conditional_c)
                # 依据贝叶斯公式计算后验概率
                posterior = prior + conditional

                posteriors.append(posterior)
            # 获取分类结果
            pred_class=self.classes[np.argmax(posteriors)]
            predictions.append(pred_class)
        return np.array(predictions)

    def Gaussian_pdf(self, cls, x):
        mean = self.params[cls]['mean']
        var = self.params[cls]['var']
        # 计算高斯分布的分子部分
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        # 计算高斯分布的分母部分
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

nb =myGaussianNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"自定义朴素贝叶斯分类器的准确率: {accuracy:.2f}")
