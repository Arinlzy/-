import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/home/xyt/lzy_temp/ML_LabCodes/Final_Lab/unemployment_rate_by_age_groups(1).csv')
# 将Year<=2017的样本作为训练集
train_df = df[df['Year'] <= 2017]
# 将Year>2017的样本作为测试集

# 
