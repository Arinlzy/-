# import numpy as np
# import matplotlib.pyplot as plt
# 
# # 设置随机种子以确保可重复性
# np.random.seed(0)
# 
# # 生成两个类别的数据
# class_1 = np.random.rand(3, 2)  # 类别1的三个样本
# class_2 = np.random.rand(3, 2) + 1.5  # 类别2的三个样本
# 
# # 绘制散点图
# plt.scatter(class_1[:, 0], class_1[:, 1], c='b', marker='o', label='Class 1')
# plt.scatter(class_2[:, 0], class_2[:, 1], c='r', marker='s', label='Class 2')
# 
# # 添加标签和图例
# plt.title('2D Dataset with 6 Samples')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# 
# # 显示图形
# plt.grid(True)
# plt.show()
float_number = 3.14
integer_part = float_number.__int__()
print(integer_part)  # 输出：3

