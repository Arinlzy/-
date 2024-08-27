# 2. CART
import time
from math import log
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#TODO 计算当前集合的Gini系数
def calcGini(dataset):
    # 求总样本数
    num_samples = len(dataset)
    # 统计各个类别的样本数量
    label_counts = {}
    for data in dataset:
        current_label = data[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # 计算Gini系数
    gini = 1.0
    for key in label_counts:
        prob = label_counts[key] / num_samples
        gini -= prob ** 2
    return gini

# 提取子集合
# 功能：从dataSet中先找到所有第axis个标签值 = value的样本
# 然后将这些样本删去第axis个标签值，再全部提取出来成为一个新的样本集
def create_sub_dataset(dataset, index, value):
    sub_dataset = []
    for example in dataset:
        current_list = []
        if example[index] == value:
            current_list = example[:index]
            current_list.extend(example[index + 1:])
            sub_dataset.append(current_list)
    return sub_dataset

# 将当前样本集分割成特征i取值为value的一部分和取值不为value的一部分（二分）
def split_dataset(dataset, index, value):
    sub_dataset1 = []
    sub_dataset2 = []
    for example in dataset:
        current_list = []
        if example[index] == value:
            current_list = example[:index]
            current_list.extend(example[index + 1:])
            sub_dataset1.append(current_list)
        else:
            current_list = example[:index]
            current_list.extend(example[index + 1:])
            sub_dataset2.append(current_list)
    return sub_dataset1, sub_dataset2

def choose_best_feature(dataset):
    # 特征总数
    numFeatures = len(dataset[0]) - 1
    # 当只有一个特征时
    if numFeatures == 1:
        return 0
    # 初始化最佳基尼系数
    bestGini = 1
    # 初始化最优特征
    index_of_best_feature = -1
    # 遍历所有特征，寻找最优特征和该特征下的最优切分点
    for i in range(numFeatures):
        # 去重，每个属性值唯一
        uniqueVals = set(example[i] for example in dataset)
        # Gini字典中的每个值代表以该值对应的键作为切分点对当前集合进行划分后的Gini系数
        Gini = {}
        # 对于当前特征的每个取值
        for value in uniqueVals:
            # 先求由该值进行划分得到的两个子集
            sub_dataset1, sub_dataset2 = split_dataset(dataset, i, value)
            #TODO 求两个子集占原集合的比例系数prob1 prob2
            prob1 = len(sub_dataset1) / len(dataset)
            prob2 = len(sub_dataset2) / len(dataset)
            # 计算子集1的Gini系数
            Gini_of_sub_dataset1 = calcGini(sub_dataset1)
            # 计算子集2的Gini系数
            Gini_of_sub_dataset2 = calcGini(sub_dataset2)
            #TODO 计算由当前最优切分点划分后的最终Gini系数
            Gini[value] = prob1 * Gini_of_sub_dataset1 + prob2 * Gini_of_sub_dataset2
            #TODO 更新最优特征和最优切分点
            if Gini[value] < bestGini:
                bestGini = Gini[value]
                index_of_best_feature = i
                best_split_point = value
    return index_of_best_feature, best_split_point

# 返回具有最多样本数的那个标签的值（'yes' or 'no'）
def find_label(classList):
    # 初始化统计各标签次数的字典
    # 键为各标签，对应的值为标签出现的次数
    labelCnt = {}
    for key in classList:
        if key not in labelCnt.keys():
            labelCnt[key] = 0
        labelCnt[key] += 1
    # 将classCount按值降序排列
    # 例如：sorted_labelCnt = {'yes': 9, 'no': 6}
    sorted_labelCnt = sorted(labelCnt.items(), key=lambda a: a[1], reverse=True)
    # 下面这种写法有问题
    # sortedClassCount = sorted(labelCnt.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 取sorted_labelCnt中第一个元素中的第一个值，即为所求
    return sorted_labelCnt[0][0]

def create_decision_tree(dataset, features):
    # 求出训练集所有样本的标签
    label_list = [example[-1] for example in dataset]
    # 先写两个递归结束的情况：
    # 若当前集合的所有样本标签相等（即样本已被分“纯”）
    # 则直接返回该标签值作为一个叶子节点
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # 若训练集的所有特征都被使用完毕，当前无可用特征，但样本仍未被分“纯”
    # 则返回所含样本最多的标签作为结果
    if len(features) == 0:
        return find_label(label_list)
    # 下面是正式建树的过程
    # 选取进行分支的最佳特征的下标和最佳切分点
    index_of_best_feature, best_split_point = choose_best_feature(dataset)
    # 得到最佳特征
    best_feature = features[index_of_best_feature]
    # 初始化决策树
    decision_tree = {best_feature: {}}
    # 使用过当前最佳特征后将其删去
    del (features[index_of_best_feature])
    # 子特征 = 当前特征（因为刚才已经删去了用过的特征）
    sub_labels = features[:]
    # 递归调用create_decision_tree去生成新节点
    # 生成由最优切分点划分出来的二分子集
    sub_dataset1, sub_dataset2 = split_dataset(dataset, index_of_best_feature, best_split_point)
    # 构造左子树
    decision_tree[best_feature][best_split_point] = create_decision_tree(sub_dataset1, sub_labels)
    # 构造右子树
    decision_tree[best_feature]['others'] = create_decision_tree(sub_dataset2, sub_labels)
    return decision_tree

# 用上面训练好的决策树对新样本分类
def predict(decision_tree, features, test_example):
    # 根节点代表的属性
    first_feature = list(decision_tree.keys())[0]
    # second_dict是第一个分类属性的值（也是字典）
    second_dict = decision_tree[first_feature]
    # 树根代表的属性，所在属性标签中的位置，即第几个属性
    index_of_first_feature = features.index(first_feature)
    # 对于second_dict中的每一个key
    for key in second_dict.keys():
        # 不等于'others'的key
        if key != 'others':
            if test_example[index_of_first_feature] == key:
                # 若当前second_dict的key的value是一个字典
                if type(second_dict[key]).__name__ == 'dict':
                    # 则需要递归查询
                    classLabel = predict(second_dict[key], features, test_example)
                # 若当前second_dict的key的value是一个单独的值
                else:
                    # 则就是要找的标签值
                    classLabel = second_dict[key]
            # 如果测试样本在当前特征的取值不等于key，就说明它在当前特征的取值属于'others'
            else:
                # 如果second_dict['others']的值是个字符串，则直接输出
                if isinstance(second_dict['others'], int):
                    classLabel = second_dict['others']
                # 如果second_dict['others']的值是个字典，则递归查询
                else:
                    classLabel = predict(second_dict['others'], features, test_example)
    return classLabel

class_num = 2  # wdbc数据集有2种labels，分别是“2,4”
attribute_len = 9  # wdbc数据集每个样本有9个属性
epsilon = 0.001  # 设定阈值

if __name__ == '__main__':
    print("Start read data...")

    time_1 = time.time()

    raw_data = pd.read_csv('breast-cancer-wisconsin.data', header=None)  # 读取csv数据
    data = raw_data.values

    features = data[:, 1:-1]
    # 删除缺失值
    index0 = np.where(features[:, 5] != '?')
    features = features[index0].astype('int32')
    labels = data[:, -1][index0]
    # 避免过拟合，采用交叉验证，随机选取33%数据作为测试集，剩余为训练集
    train_attributes, test_attributes, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                    random_state=0)
    train_data = (np.concatenate((train_attributes,train_labels[:,np.newaxis]),axis=1)).tolist()
    time_2 = time.time()
    #创建决策树
    dicision_Tree = create_decision_tree(train_data,list(range(attribute_len)))
    #测试数据
    test_predict = []
    for value in test_attributes:
        test_predict.append(predict(dicision_Tree,list(range(attribute_len)),value))
    score = accuracy_score(test_labels.astype('int32'), test_predict)
    print("The accruacy score is %f" % score)
