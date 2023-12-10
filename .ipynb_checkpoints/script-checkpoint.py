import numpy as np
import matplotlib.pyplot as plt
import tsfel
from scipy.signal import argrelextrema, find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import math
import json
from sklearn.metrics import mean_squared_error

# 用来去除peaks里的错误值
def update_array(a, data_tmp):
    i = 0
    while i < len(a) - 2:
        if data_tmp[a[i]] < data_tmp[a[i + 1]] < data_tmp[a[i + 2]]:
            a = np.delete(a, i)
        elif data_tmp[a[i]] > data_tmp[a[i + 1]] > data_tmp[a[i + 2]]:
            a = np.delete(a, i + 2)
        else:
            i += 1
    return a

# 用来获得信号的峰值点
def get_peaks(signal):
    t = np.arange(1000)
    # 峰值检测
    peak_indices, _ = find_peaks(signal)  # 返回极大值点的索引

    # 线性插值
    t_peaks = t[peak_indices]  # 极大值点的时间
    peak_values = signal[peak_indices]  # 极大值点的幅值
    interpolation_func = interp1d(t_peaks, peak_values, kind='linear', bounds_error=False, fill_value=0)
    envelope = interpolation_func(t)
    
    # 提取峰值点形成的波的波峰
    peaks2,_ = find_peaks(envelope, distance = 10)

    peaks2 = update_array(peaks2, signal)
    
    if signal[peaks2[0]]>signal[peaks2[1]]:
        peaks2 = np.delete(peaks2, 0)
    
    if len(peaks2)%2 != 0:
        peaks2 = np.delete(peaks2, len(peaks2) - 1)
    
    return peaks2

# use cluster method to get the template
def get_template(signal):

    peaks2 = get_peaks(signal)

    avg_index = (peaks2[::2] + peaks2[1::2]) // 2

    # 使用这些平均数作为x的下标，将x切割成多个部分
    splits = np.split(signal, avg_index)

    max_length = max(len(split) for split in splits)

    # 补充每个部分使其长度相等
    padded_splits = [np.pad(split, (0, max_length - len(split))) for split in splits]

    # 将这些部分堆叠成一个二维数组
    stacked_array = np.vstack(padded_splits)
    stacked_array = np.delete(stacked_array, 0, axis=0)

    class PulseClustering:
        def __init__(self, threshold):
            self.threshold = threshold
            self.clusters = []

        def fit(self, pulses):
            for pulse in pulses:
                if not self.clusters:  # 如果聚类为空，创建第一个聚类
                    self.clusters.append([pulse])
                else:
                    for cluster in self.clusters:
                        center_pulse = np.mean(cluster, axis=0)  # 计算聚类中心
                        rmse = np.sqrt(mean_squared_error(center_pulse, pulse))  # 计算RMSE
                        if rmse < self.threshold:  # 如果RMSE低于阈值，将脉冲添加到聚类中
                            cluster.append(pulse)
                            break
                    else:  # 如果脉冲与现有的所有聚类的中心的RMSE都高于阈值，创建新的聚类
                        self.clusters.append([pulse])

        def get_clusters(self):
            return self.clusters

    threshold = 0.000005  # 这是一个选择的阈值

    clustering = PulseClustering(threshold)
    clustering.fit(stacked_array)
    clusters = clustering.get_clusters()

    num_pulses_per_cluster = [len(cluster) for cluster in clusters]

    # 打印结果
#     for i, num_pulses in enumerate(num_pulses_per_cluster):
#         print(f"Cluster {i+1} contains {num_pulses} pulses.")

    max_cluster = max(clusters, key=len)

    # 计算最大聚类的平均脉冲
    average_pulse = np.mean(max_cluster, axis=0)
    return average_pulse

data_train = np.load("./data/simu_20000_0.1_90_140_train.npy")
data_test = np.load("./data/simu_10000_0.1_141_178_test.npy")


# 获得train数据的48个TSFEL特征
with open('./all_features.json', 'r') as file:
    cfg_file = json.load(file)

feature_48TSFEL_train = np.zeros((1,48))
for i in range(data_train.shape[0]):
    signal = data_train[i, :1000]
    template = get_template(signal)
    
    features = tsfel.time_series_features_extractor(cfg_file, template, fs=100, window_size=len(template)).values.flatten()

    feature_48TSFEL_train = np.vstack((feature_48TSFEL_train, features))

feature_48TSFEL_train = np.delete(feature_48TSFEL_train, 0, axis = 0)
np.save("./data/feature_48TSFEL_train", feature_48TSFEL_train)


# 获得test数据的48个TSFEL特征
feature_48TSFEL_test = np.zeros((1,48))

for i in range(data_test.shape[0]):
    signal = data_test[i, :1000]
    template = get_template(signal)
    
    features = tsfel.time_series_features_extractor(cfg_file, template, fs=100, window_size=len(template)).values.flatten()

    feature_48TSFEL_test = np.vstack((feature_48TSFEL_test, features))

feature_48TSFEL_test = np.delete(feature_48TSFEL_test, 0, axis = 0)
np.save("./data/feature_48TSFEL_test", feature_48TSFEL_test)