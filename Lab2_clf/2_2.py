from calendar import c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
np.random.seed(0)

# 生成两个类别的数据
class_1 = np.random.rand(3, 2)  # 类别1的三个样本
class_2 = np.random.rand(3, 2) + 1.5  # 类别2的三个样本
data = np.concatenate((class_1, class_2), axis=0)
# print(data)
# print(type(data[0][1]))

X = data
y = np.array([1, 1, 1, -1, -1, -1])


class Model:
    def __init__(self):  # 初始化数据
        self.w = np.ones(2, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        # self.data = data

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # 随机梯度下降法
    def fit(self, X_train, y_train):
        is_wrong = False
        epoc = 0
        while not is_wrong:
            epoc += 1
            wrong_count = 0  # 初始设置错误次数为0
            print()
            print(f"------------第{epoc}轮训练------------")
            for d in range(len(X_train)):
                print(f"  ---------对第{d}个数据---------")
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    # 权重self.w和截距self.b更新
                    # TODO XXX
                    print(f"修正前: w = {self.w}, b = {self.b}")
                    self.w = self.w + self.l_rate * X * y
                    # TODO XXX
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
                    print(f"修正后: w = {self.w}, b = {self.b}")
                    print(f"w变化：{self.l_rate * X * y}")
                    print(f"b变化：{self.l_rate * y}")
                    self.show()
                else:
                    print(f"正确分类")

            if wrong_count == 0:  # 误分点数目为0跳出循环
                is_wrong = True
        return 'Perceptron Model!'

    def score(self):
        pass

    def show(self):
        x_points = np.linspace(0, 3, 10)
        y_ = -(self.w[0] * x_points + self.b) / self.w[1]
        plt.plot(x_points, y_)
        plt.plot(data[0:3, 0], data[0:3, 1], 'bo', color='blue', label='0')
        plt.plot(data[3:, 0], data[3:, 1], 'bo', color='orange', label='1')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()


perceptron = Model()
perceptron.fit(X, y)
