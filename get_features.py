import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis

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

def delete_unique(a):
    mean = np.mean(a)
    std = np.std(a)
    # 设置阈值
    threshold = 1
    # 使用布尔索引删除特殊值
    filtered_a = a[np.abs(a - mean) <= threshold * std]
    return filtered_a

def get_mag_0(peaks2, data_tmp):
    return np.mean(data_tmp[peaks2[::2]])

def get_mag_1(peaks2, data_tmp):
    return np.mean(data_tmp[peaks2[1::2]])

def get_mag_0_max(peaks2, data_tmp):
    return np.max(data_tmp[peaks2[::2]])

def get_mag_1_max(peaks2, data_tmp):
    return np.max(data_tmp[peaks2[1::2]])

def get_mag_0_min(peaks2, data_tmp):
    return np.min(data_tmp[peaks2[::2]])

def get_mag_1_min(peaks2, data_tmp):
    return np.min(data_tmp[peaks2[1::2]])

def get_features(data_tmp):
    max = np.max(data_tmp)
    min = np.min(data_tmp)
    data_tmp_raw = data_tmp
    data_tmp = data_tmp / max

    fs = 100
    t = np.linspace(0, 10, 10 * fs)
    signal = data_tmp

    # 峰值检测
    peak_indices, _ = find_peaks(signal)  # 返回极大值点的索引

    # 线性插值
    t_peaks = t[peak_indices]  # 极大值点的时间
    peak_values = signal[peak_indices]  # 极大值点的幅值
    interpolation_func = interp1d(t_peaks, peak_values, kind='linear', bounds_error=False, fill_value=0)
    envelope = interpolation_func(t)

    peaks2, _ = find_peaks(envelope, distance=10)

    peaks2 = update_array(peaks2, data_tmp)
    if len(peaks2) % 2 != 0:
        peaks2 = np.delete(peaks2, len(peaks2) - 1)

    diff_12 = peaks2[1::2] - peaks2[0::2]
    diff_21 = peaks2[2::2] - peaks2[1:-1:2]
    diff_22 = peaks2[2::2] - peaks2[:-2:2]

    # diff_12 = delete_unique(diff_12)
    # diff_21 = delete_unique(diff_21)
    # diff_22 = delete_unique(diff_22)

    
    diff_12_mean = np.mean(diff_12)
    diff_21_mean = np.mean(diff_21)

    if diff_12_mean < diff_21_mean:
        diff_min = diff_12_mean
    else:
        diff_min = diff_21_mean

    diff_min = int(diff_min / 3)    
        
    m_12 = np.mean(diff_12)
    m_21 = np.mean(diff_21)
    m_22 = np.mean(diff_22)

    signal2 = -1 * signal

    tau = diff_min * 3 / 4

    peaks2_down = []

    for i in range(1, len(peaks2)):
        peaks2_down.append(int(peaks2[i] - tau)  + np.argmax(signal2[int(peaks2[i] - tau):int(peaks2[i] + tau)]))


    mag_0_down = get_mag_1(peaks2_down, data_tmp_raw)
    mag_1_down = get_mag_0(peaks2_down, data_tmp_raw)
    # print(mag_0, mag_1)
    
    
    mag_0 = get_mag_0(peaks2, data_tmp_raw)
    mag_1 = get_mag_1(peaks2, data_tmp_raw)
    mag_0_max = get_mag_0_max(peaks2, data_tmp_raw)
    mag_1_max = get_mag_1_max(peaks2, data_tmp_raw)
    mag_0_min = get_mag_0_min(peaks2, data_tmp_raw)
    mag_1_min = get_mag_1_min(peaks2, data_tmp_raw)


    kurt = []
    sk = []
    
    
    for i in range(1, len(peaks2) - 1):
        seg = data_tmp[peaks2[i] - diff_min:peaks2[i] + diff_min]
        if len(seg) > 0:
            kurt.append(kurtosis(seg))
            sk.append(skew(seg))

            
    kurt2 = kurt[::2]
    kurt1 = kurt[1::2]

    kurt2_mean = np.mean(kurt2)
    kurt1_mean = np.mean(kurt1)
    
    skew2 = sk[::2]
    skew1 = sk[1::2]
    
    skew2_mean = np.mean(skew2)
    skew1_mean = np.mean(skew1)

    
    # m_12, m_21------------12
    # m_12, m_21, m_22, mag_0, mag_1, mag_0 / mag_1------------12
    # kurt1_mean, kurt2_mean----------------22
    # m_12, m_21, kurt1_mean, kurt2_mean-------------------9.3
    # kurt2_mean, kurt1_mean, skew1_mean, skew2_mean----------------20
    # m_12, m_21, kurt2_mean, kurt1_mean, skew1_mean, skew2_mean----------------9.0
    # m_12, kurt2_mean, kurt1_mean, skew1_mean, skew2_mean----------------13.7 waiting
    # m_12, m_21, skew1_mean, skew2_mean----------------9.6
    # m_12, m_21, kurt1_mean----------------10.0
    # m_12, m_21, kurt2_mean----------------10.2
    # m_12, m_21, skew1_mean---------------9.8
    # m_12, m_21, skew2_mean---------------10.9
    
    
    features = np.array([m_12, m_21, mag_0, mag_1, mag_0 + mag_0_down, mag_1 + mag_1_down, mag_0 + mag_0_down - mag_1 - mag_1_down, mag_0 - mag_0_down, mag_1 - mag_1_down,kurt2_mean, kurt1_mean, skew1_mean, skew2_mean])

    # features = np.array([m_12, m_21, m_22, mag_0, mag_1, mag_0 - mag_1])

    return features



if __name__ == "__main__":
    data_train = np.load("./data/simu_20000_0.1_90_140_train.npy")
    features = get_features(data_train[0, :1000])
    for i in range(1, 20000):
        tmp = get_features(data_train[i, :1000])
        features = np.vstack((features, tmp))
    np.save("./data/features_train.npy", features)

    data_train = np.load("./data/simu_10000_0.1_141_178_test.npy")
    features = get_features(data_train[0, :1000])
    for i in range(1, 10000):
        tmp = get_features(data_train[i, :1000])
        features = np.vstack((features, tmp))
    np.save("./data/features_test.npy", features)

    scaler = MinMaxScaler()

    tmp = np.load("./data/features_train.npy")
    scaler.fit(tmp)
    tmp = scaler.transform(tmp)
    np.save("./data/features_train_norm.npy", tmp)

    tmp = np.load("./data/features_test.npy")
    tmp = scaler.transform(tmp)
    np.save("./data/features_test_norm.npy", tmp)