from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from pandas import DataFrame
import numpy as np
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
df = DataFrame(iris.data, columns=iris.feature_names)
df['target'] = list(iris.target)
X = df.iloc[:, 0:4]
Y = df.iloc[:, 4]
# 划分数据
# 这里可以使train_test_spilt快速切分，但是数据分布会不均匀，我这里手动切分
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
# 将每一类别的前80%作为训练集，后20%作为测试集
train_indices = list(range(0, 40)) + list(range(50, 90)) + list(range(100, 140))
test_indices = list(range(40, 50)) + list(range(90, 100)) + list(range(140, 150))
X_train = df.iloc[train_indices, :4].reset_index(drop=True)
Y_train = df.iloc[train_indices, 4].reset_index(drop=True)
X_test = df.iloc[test_indices, :4].reset_index(drop=True)
Y_test = df.iloc[test_indices, 4].reset_index(drop=True)

sc = StandardScaler()
sc.fit(X)
standard_train = sc.transform(X_train)
standard_test = sc.transform(X_test)

# python调用
# 构建mlp模型
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', random_state=1)

# 拟合数据
mlp.fit(X_train, Y_train)

# 得到预测结果
result = mlp.predict(X_test)


# 查看模型结果
print("测试集合的y值：", list(Y_test))
print("神经网络预测的的y值：", list(result))
print("预测的准确率为：", mlp.score(standard_test, Y_test))
print("层数为：", mlp.n_layers_)
print("迭代次数为：", mlp.n_iter_)
print("损失为：", mlp.loss_)
print("激活函数为：", mlp.out_activation_)

# 手动实现

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重
        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(
            self.hidden_size, self.output_size)

        # 初始化偏置
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # sigmoid计算方式

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # sigmoid导数计算方式

    def forward(self, X):
        # 计算隐藏层的输入
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden #
        # 计算隐藏层的输出
        self.hidden_output = self.sigmoid(self.hidden_input)
        # 计算输出层的输入
        self.output_input = np.dot(
            self.hidden_output, self.weights_hidden_output) + self.bias_output
        # 计算输出层的输出
        self.output = self.sigmoid(self.output_input)
        return self.output

    def backward(self, X, y, output, learning_rate):
        # 计算输出层的误差
        output_error = y - output
        # 计算输出层的梯度
        output_delta = output_error * self.sigmoid_derivative(output) #
        # 计算隐藏层的误差
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        # 计算隐藏层的梯度
        hidden_delta = hidden_error * \
            self.sigmoid_derivative(self.hidden_output)

        # 更新权重和偏置
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, output_delta) #
        self.bias_output += learning_rate * \
            np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta)
        self.bias_hidden += learning_rate * \
            np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(0.5 * (y - output) ** 2)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

    def predict(self, X):
        return np.round(self.forward(X))


# 将标签转换为独热编码
def one_hot_encode(labels):
    num_classes = len(np.unique(labels))
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
    return one_hot_labels

# def accuracy_score(y_true, y_pred):
#     # 确保标签长度一致
#     if len(y_true) != len(y_pred):
#         raise ValueError("y_true 和 y_pred 长度不一致")
# 
#     # 计算正确预测的数量
#     correct = 0
#     for true_label, pred_label in zip(y_true, y_pred):
#         if true_label == pred_label:
#             correct += 1
# 
#     # 计算准确率
#     accuracy = correct / len(y_true)
#     return accuracy


# 构建神经网络
input_size = X_train.shape[1]
hidden_size = 10
output_size = len(np.unique(Y_train))  # 根据训练集标签确定输出层大小
nn = NeuralNetwork(input_size, hidden_size, output_size)

# 将标签转换为独热编码
Y_train_encoded = one_hot_encode(Y_train)

# 训练神经网络
print('training.......')
nn.train(standard_train, Y_train_encoded, epochs=1000, learning_rate=0.1)

# 预测测试集
predictions = nn.predict(standard_test)

# 计算准确率
accuracy = accuracy_score(Y_test, np.argmax(predictions, axis=1))

# 查看模型结果
print("测试集合的y值：", list(Y_test))
print("神经网络预测的的y值：", list(np.argmax(predictions, axis=1)))
print("预测的准确率为：", accuracy)
