from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from Multi_AdaBoost import AdaBoostClassifier

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
# 划分数据集，80%的训练数据，20%的测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('数据集样本数：{},训练样本数：{},测试集样本数：{}'.format(len(X),len(X_train),len(X_test)))


# 构造一个AdaBoost分类器并拟合训练数据，要求：
	# 使用的算法为SAMME.R
	# 弱分类器为DecisionTreeClassifier，最大深度为2
	# 弱分类器数量为4
	# 学习率为1
SAMMER = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
                            n_estimators=4,
                            learning_rate=1,
                            algorithm='SAMME.R',
                            random_state=None)
SAMMER.fit(X_train, y_train)

# 构造一个AdaBoost分类器并拟合训练数据，要求：
	# 使用的算法为SAMME
	# 弱分类器为DecisionTreeClassifier，最大深度为2
	# 弱分类器数量为4
	# 学习率为1
SAMME = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
                           n_estimators=4,
                           learning_rate=1,
                           algorithm='SAMME',
                           random_state=None)
SAMME.fit(X_train, y_train)

print("SAMME")
y_pred = SAMME.predict(X_test)
print("精确率",precision_score(y_test, y_pred, average='weighted'))
print("召回率",recall_score(y_test, y_pred, average='weighted'))
print("F1度量值",f1_score(y_test, y_pred, average='weighted'))

print("SAMME.R")
r_y_pred = SAMMER.predict(X_test)
print("精确率",precision_score(y_test, r_y_pred, average='weighted'))
print("召回率",recall_score(y_test, r_y_pred, average='weighted'))
print("F1度量值",f1_score(y_test, r_y_pred, average='weighted'))
