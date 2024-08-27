# encoding=utf-8
import copy
import time
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Tree(object):
    def __init__(self,node_type,Class = None, attribute = None):
        self.node_type = node_type  # 节点类型（internal或leaf）
        self.dict = {} # dict的键表示属性Ag的可能值ai，值表示根据ai得到的子树
        self.Class = Class  # 叶节点表示的类，若是内部节点则为none
        self.attribute = attribute # 表示当前的树即将由第attribute个属性划分（即第attribute属性是使得当前树中信息增益最大的属性）

    def add_tree(self,key,tree):
        self.dict[key] = tree

    def predict(self,attributes):
        #print('attribute', self.attribute)
        if self.node_type == 'leaf' or (attributes[self.attribute] not in self.dict):
            #print('self.Class', self.Class)
            return self.Class

        tree = self.dict.get(attributes[self.attribute])
        return tree.predict(attributes)

# 计算数据集x的经验熵H(x)
def calc_ent(x):
    # 统计每个类别出现的次数
    label_count = Counter(x)
    # 计算经验熵
    ent = 0.0
    for key in label_count:
        prob = label_count[key] / len(x)
        ent -= prob * np.log2(prob)
    return ent

# 计算条件熵H(y/x)
def calc_condition_ent(x, y):
    # 统计x中每个取值对应的y的类别
    label_x = Counter(x)
    # 计算条件熵
    condition_ent = 0.0
    for key in label_x:
        # 选取x取值为key的样本
        sub_y = y[np.where(x == key)[0]]
        # 计算该取值下y的经验熵
        ent = calc_ent(sub_y)
        condition_ent += label_x[key] / len(x) * ent
    return condition_ent

# 计算信息增益
def calc_ent_gain(x,y):
    # 计算数据集y的经验熵（即父节点的熵）
    ent = calc_ent(y)   
    # 计算条件熵H(y/x)（即在给定属性x的条件下，y的熵）
    condition_ent = calc_condition_ent(x, y)
    # 计算信息增益
    ent_gain = ent - condition_ent
    return ent_gain

# ID3算法
def recurse_train_ID3(train_set,train_label,attributes):
    LEAF = 'leaf'
    INTERNAL = 'internal'

    # 步骤1——如果训练集train_set中的所有实例都属于同一类C, 则将该类表示为新的叶子节点
    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(LEAF, Class=label_set.pop())

    # 步骤2——如果属性集为空，表示不能再分了，将剩余样本中样本数最多的一类赋给叶子节点
    class_len = [(i, len(list(filter(lambda x: x == i, train_label)))) for i in label_set]  # 计算每一个类出现的个数
    (max_class, max_len) = max(class_len, key=lambda x: x[1])  # 出现个数最多的类

    if len(attributes) == 0:
        return Tree(LEAF, Class=max_class)

    # 步骤3——计算信息增益,并选择信息增益最大的属性
    max_attribute = 0
    max_gain = 0
    D = train_label
    for attribute in attributes:
        # print(type(train_set))
        A = np.array(train_set[:,attribute].flat) # 选择训练集中的第attribute列（即第attribute个属性）
        #计算信息增益并更新max_gain和max_attribute
        gain = calc_ent_gain(A, D)
        if gain > max_gain:
            max_gain = gain
            max_attribute = attribute

    # 步骤4——依据样本在最大增益属性的取值，划分非空子集，进而构建树或子树
    sub_attributes = list(filter(lambda x:x!=max_attribute,attributes))
    tree = Tree(INTERNAL,attribute=max_attribute)

    max_attribute_col = np.array(train_set[:,max_attribute].flat)
    attribute_value_list = set([max_attribute_col[i] for i in range(max_attribute_col.shape[0])]) # 保存信息增益最大的属性可能的取值 (shape[0]表示计算行数)
    for attribute_value in attribute_value_list:

        index = list(np.where(train_set[:,max_attribute] == attribute_value)[0])

        sub_train_set = train_set[index]
        sub_train_label = train_label[index]
        #递归建树
        sub_tree = recurse_train_ID3(sub_train_set, sub_train_label, sub_attributes)
        tree.add_tree(attribute_value, sub_tree)

    return tree


# C4.5算法
def recurse_train_C45(train_set,train_label,attributes):

    LEAF = 'leaf'
    INTERNAL = 'internal'
    # 步骤1——如果训练集train_set中的所有实例都属于同一类C, 则将该类表示为新的叶子节点
    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(LEAF,Class = label_set.pop())

    # 步骤2——如果属性集为空，表示不能再分了，将剩余样本中样本数最多的一类赋给叶子节点
    class_len = [(i,len(list(filter(lambda x:x==i,train_label)))) for i in label_set] # 计算每一个类出现的个数
    (max_class,max_len) = max(class_len,key = lambda x:x[1])  #出现个数最多的类

    if len(attributes) == 0:
        return Tree(LEAF,Class = max_class)
    # 步骤3——计算信息增益率,并选择信息增益率最大的属性
    max_attribute = 0
    max_gain_r = 0
    D = train_label
    for attribute in attributes:
        A = np.array(train_set[:,attribute].flat) # 选择训练集中的第attribute列（即第attribute个属性）
        gain = calc_ent_gain(A, D)
        ####### 计算信息增益率，并更新max_attribute，max_gain_r
        
        split_info = calc_ent(A) # 计算分裂信息
        if split_info == 0:  # 避免分母为0
            continue
        gain_ratio = gain / split_info
        if gain_ratio > max_gain_r:
            max_gain_r = gain_ratio
            max_attribute = attribute
            
    # 步骤4——如果最大的信息增益率小于阈值,说明所有属性的增益都非常小，那么取样本中最多的类为叶子节点
    if max_gain_r < epsilon:
        return Tree(LEAF,Class = max_class)

    # 步骤5——依据样本在最大增益率属性的取值，划分非空子集，进而构建树或子树
    sub_attributes = list(filter(lambda x:x!=max_attribute,attributes))
    tree = Tree(INTERNAL,attribute=max_attribute)
    max_attribute_col = np.array(train_set[:,max_attribute].flat)
    attribute_value_list = set([max_attribute_col[i] for i in range(max_attribute_col.shape[0])]) # 保存信息增益最大的属性可能的取值 (shape[0]表示计算行数)
    for attribute_value in attribute_value_list:

        index = list(np.where(train_set[:,max_attribute] == attribute_value)[0])

        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        #递归建树
        sub_tree = recurse_train_C45(sub_train_set, sub_train_label, sub_attributes)
        tree.add_tree(attribute_value, sub_tree)
    return tree

def train(train_set,train_label,attributes):
    # return recurse_train_ID3(train_set,train_label,attributes)
    return recurse_train_C45(train_set,train_label,attributes)

def predict(test_set,tree):
    result = []

    for attributes in test_set:
        tmp_predict = tree.predict(attributes)
        if tmp_predict == None:
            tmp_predict = epsilon
        result.append(tmp_predict)

    return np.array(result)

def post_pruning(val_data, val_label, dataset, tree, features):
    path = []
    DFS.val_data = val_data
    DFS.val_label = val_label
    DFS.dataset = dataset
    DFS.root = tree
    DFS.features = features
    DFS(tree, path)
    return tree

def DFS(tree, path):
    if tree.node_type == "leaf":
        return

    # 没有return，该节点是内部节点，检测该节点的子节点是不是都是叶子节点
    leafs_parent = True
    for sub_tree in tree.dict.values():
        if sub_tree.node_type != "leaf":
            leafs_parent = False
            break
    # 如果该节点的子节点全部是叶子节点
    if leafs_parent and len(path) > 0:
        process_leafs_parent(DFS.val_data, DFS.val_label,
                             DFS.dataset, DFS.root, DFS.features, tree, path)

    # 如果该节点的子节点也是内部节点，继续递归
    else:
        for key, sub_tree in tree.dict.items():
            path.append((DFS.features[tree.attribute], key))
            DFS(sub_tree, path)
            path.pop()

        # 对子节点的递归结束，
        # 再次检测该节点的子节点是不是都是叶子节点
        leafs_parent = True
        for sub_tree in tree.dict.values():
            if sub_tree.node_type != "leaf":
                leafs_parent = False
                break
        # 如果该节点的子节点全部是叶子节点
        if leafs_parent and len(path) > 0:
            process_leafs_parent(DFS.val_data, DFS.val_label,
                                 DFS.dataset, DFS.root, DFS.features, tree, path)


def process_leafs_parent(val_data, val_label,
                         dataset, root, features, tree, path):
    # 在未剪枝的决策树上进行验证，并得出准确率w
    old_score = accuracy_score(val_label.astype('int32'), predict(val_data, root).astype('int32'))
    # 将未剪枝的子树保存起来
    temp_tree = copy.deepcopy(tree)

    # 根据路径找到训练集中能到达该节点的数据
    df = pd.DataFrame(dataset, columns=features + ['label'])
    df = df[df[path[0][0]]==path[0][1]]
    if len(path) >= 2:
        for each in path[1:]:
            df = df[df[each[0]]==each[1]]

    # 进行剪枝
    class_count = Counter(df['label'])
    majority_class = class_count.most_common(1)[0][0]  # 取得样本数最多的类别
    tree.Class = majority_class
    tree.node_type = 'leaf'
    tree.dict = {}

    # 若剪枝后分数变低，那么还原子树
    new_score = accuracy_score(val_label.astype('int32'), predict(val_data, root).astype('int32'))
    if (new_score < old_score):
        tree.Class = temp_tree.Class
        tree.node_type = temp_tree.node_type
        tree.dict = temp_tree.dict

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
    index0 = np.where(features[:,5]!='?')
    features = features[index0].astype('int32')
    labels = data[:,-1][index0]

    # 避免过拟合，采用交叉验证，随机选取33%数据作为测试集，剩余为训练集
    train_attributes, test_attributes, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)

    # 通过C4.5算法生成决策树
    print('Start training...')
    tree = train(train_attributes,train_labels,list(range(attribute_len)))

    print('Start predicting...')
    test_predict = predict(test_attributes,tree)
    print(test_predict)
    score = accuracy_score(test_labels.astype('int32'), test_predict.astype('int32'))
    print("The accruacy score is %f" % score)
    #剪枝
    new_tree = post_pruning(test_attributes, test_labels, np.c_[train_attributes,train_labels], tree, list(range(attribute_len)))
    new_test_predict = predict(test_attributes,new_tree)
    for i in range(len(test_predict)):
        if new_test_predict[i] == None:
            new_test_predict[i] = epsilon
    score = accuracy_score(test_labels.astype('int32'), new_test_predict.astype('int32'))
    print("The new accruacy score is %f" % score)

