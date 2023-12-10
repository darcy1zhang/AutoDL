import numpy as np
import pandas as pd


# 假设您已经有了一个包含1000个样本和10个特征的.npy文件和一个1000*1的numpy数组作为目标变量
# 我们将使用随机数据作为示例
features = np.load("./data/features_test_norm.npy")
target = np.load("./data/simu_20000_0.1_90_140_train.npy")[:, 1004]

# 将特征和目标变量合并为一个DataFrame
df = pd.DataFrame(features, columns=[f'feature{i}' for i in range(1, 11)])
df['target'] = target

# 计算特征与目标变量之间的Pearson相关系数
correlation_matrix = df.corr(method='pearson')

# 提取目标变量与其他特征之间的相关系数
target_correlation = correlation_matrix['target'].drop('target')

# 按相关系数大小降序排列特征
sorted_features = target_correlation.abs().sort_values(ascending=False)

# 输出相关系数评估结果
print("Feature importance based on Pearson correlation:")
print(sorted_features)
