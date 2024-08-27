import numpy as np
from copy import deepcopy

class AdaBoostClassifier(object):
    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''AdaBoostClassifier can only be called with keyword
                   arguments for the following keywords: base_estimator ,n_estimators,
                    learning_rate,algorithm,random_state''')
        allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state']
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling")

        n_estimators = 50
        learning_rate = 1
        algorithm = 'SAMME.R'
        random_state = None

        if kwargs and not args:
            if 'base_estimator' in kwargs:
                base_estimator = kwargs.pop('base_estimator')
            else:
                raise ValueError('''base_estimator can not be None''')
            if 'n_estimators' in kwargs: n_estimators = kwargs.pop('n_estimators')
            if 'learning_rate' in kwargs: learning_rate = kwargs.pop('learning_rate')
            if 'algorithm' in kwargs: algorithm = kwargs.pop('algorithm')
            if 'random_state' in kwargs: random_state = kwargs.pop('random_state')
        # 弱评估器
        self.base_estimator_ = base_estimator
        # 集成算法中弱评估器的数量
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        # 采用的集成学习策略
        self.algorithm_ = algorithm
        # 控制每次建立决策树之前随机抽样过程的随机数种子
        self.random_state_ = random_state
        self.estimators_ = list()
        # 每个弱评估器的权重
        self.estimator_weights_ = np.zeros(self.n_estimators_)
        # 每个弱评估器的错误率
        self.estimator_errors_ = np.ones(self.n_estimators_)

    def _samme_proba(self, estimator, n_classes, X): # 计算h(x)
        proba = estimator.predict_proba(X)
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)
        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                  * log_proba.sum(axis=1)[:, np.newaxis])

    def fit(self, X, y):
        self.n_samples = X.shape[0]
        self.classes_ = np.array(sorted(list(set(y))))
        self.n_classes_ = len(self.classes_)
        for iboost in range(self.n_estimators_):
            if iboost == 0: # 初始化权重为1/n
                sample_weight = np.ones(self.n_samples) / self.n_samples
            # 循环迭代弱分类器
            sample_weight, estimator_weight, estimator_error = self.boost(X, y, sample_weight)
            # 提前结束，训练失败
            if estimator_error == None:
                break
            # 存储弱分类器权重
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight
            if estimator_error <= 0:
                break
        return self

    def boost(self, X, y, sample_weight):
        if self.algorithm_ == 'SAMME':
            return self.boost_SAMME(X, y, sample_weight)
        elif self.algorithm_ == 'SAMME.R':
            return self.boost_SAMMER(X, y, sample_weight)

    def boost_SAMMER(self, X, y, sample_weight): # SAMME.R
        estimator = deepcopy(self.base_estimator_)
        if self.random_state_:
            estimator.set_params(random_state=1)
        # 训练弱分类器
        estimator.fit(X, y, sample_weight=sample_weight)
        # 计算错误率
        y_pred = estimator.predict(X)
        incorrect = y_pred != y
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        
        # 比随机猜还差，抛弃
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            return None, None, None
            
        # 计算h(x)
        y_predict_proba = estimator.predict_proba(X)
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps
        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        y_coding = y_codes.take(self.classes_ == y[:, np.newaxis])
        
        # 更新样本权重
        intermediate_variable = (-1. * self.learning_rate_ * (((self.n_classes_ - 1) / self.n_classes_) * estimator_error))
        sample_weight *= np.exp(intermediate_variable)
        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None
        # 归一化权重
        sample_weight /= sample_weight_sum
        
        # 存储当前弱分类器
        self.estimators_.append(estimator)
        return sample_weight, 1, estimator_error

    def boost_SAMME(self, X, y, sample_weight): # SAMME
        estimator = deepcopy(self.base_estimator_)
        if self.random_state_:
            estimator.set_params(random_state=1)

        # 训练基分类器，计算结果
        estimator.fit(X, y, sample_weight=sample_weight)
        y_pred = estimator.predict(X)
        incorrect = y_pred != y
        
        # 计算错误率
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        
        # 分类效果比随机数还差，抛弃这种情况
        if estimator_error >= 1 - 1 / self.n_classes_:
            return None, None, None
            
        # 计算当前分类器权重
        estimator_weight = self.learning_rate_ * np.log((1 - estimator_error) / estimator_error)
        
        # 权重为负，无意义，抛弃
        if estimator_weight <= 0:
            return None, None, None
            
        # 更新样本权重
        sample_weight *= np.exp(estimator_weight * incorrect)
        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None
            
        # 归一化权重
        sample_weight /= sample_weight_sum
        
        # 存储当前弱分类器
        self.estimators_.append(estimator)
        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        if self.algorithm_ == 'SAMME.R':
            # SAMME.R权重均为1
            pred = sum(self._samme_proba(estimator, n_classes, X) for estimator in self.estimators_)
        else:  # SAMME
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))
        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def predict_proba(self, X):
        if self.algorithm_ == 'SAMME.R':
            # SAMME.R权重均为1
            proba = sum(self._samme_proba(estimator, self.n_classes_, X)
                        for estimator in self.estimators_)
        else:  # SAMME
            proba = sum(estimator.predict_proba(X) * w
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))
        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (self.n_classes_ - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer
        return proba
