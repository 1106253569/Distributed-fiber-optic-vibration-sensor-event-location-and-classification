import json
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from data_information import data_information


def normalization(data):  # 归一化处理
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def myfft(data):  # 快速傅里叶变换
    fft_x = fft(data)
    N = np.size(data)
    # 取复数的绝对值，即复数的模(双边频谱)
    abs_y = np.abs(fft_x)  
    # 由于对称性，只取一半区间（单边频谱）
    normalization_half_y = abs_y[range(int(N / 2))]
    return normalization_half_y


def get_row(sht, min_col: int, max_col: int, begin_row: int=1, rows_end: int = 32):
    row_datas = []
    for datas in sht.iter_rows(
        min_row=begin_row, 
        max_row=rows_end,
        min_col=min_col,
        max_col=max_col,
    ):
        temp = []
        for i in range(len(datas)):
            value = datas[i].value
            temp.append(value)
        row_datas.append(temp)
    return row_datas


def get_event_json(file_path):
    # 第一个元素为事件开始列，第二个元素为事件结束列
    with open(file_path, "r", encoding="utf-8") as json_file:
        event = json.load(json_file)
    event_label = event["label"][0]
    all_event_information = event["data"]
    return all_event_information, event_label


def get_split_data(original_dataset, step_number: int = 16):
    # 对每一个数据块元素进行分割处理
    datasets = []
    max_row = np.size(original_dataset, axis=0)
    for i in range(0, max_row, step_number):
        if i + 32 > max_row:
            break
        datasets.append(original_dataset[i : i + 32])  # type: ignore
    results = []
    for i in range(len(datasets)):
        X = datasets[i]
        temp = np.zeros(16 * 22)
        temp = temp.reshape(16, 22)
        X_fft = []
        for j in range(np.size(X, axis=1)):
            X_fft.append(myfft(X[:, j]))
        X_fft = np.array(X_fft).T
        # 选择0到21列
        loc = 0
        limit = [0, 21]
        temp[:, limit[0]] = X_fft[:, loc]
        # 为了防止列数过少出现错误
        # 因此若是所需列数超过已有列数
        # 超出部分由从开头再次选取一部分拼接在后
        l = 0
        if loc + limit[1] > np.size(X_fft, axis=1) - 1:
            l = (loc + limit[1]) - np.size(X_fft, axis=1) + 1
            temp[:, -limit[1] :] = np.append(X_fft[:, loc + 1 :],
                                             X_fft[:, :l],
                                             axis=1)
        else:
            temp[:, -limit[1] :] = X_fft[:, loc + 1 : loc + limit[1] + 1]
        # 将最终数据块转换成352*1的矩阵插入results
        results.append(temp.reshape(16 * 22, 1))
    return results


def get_YN():
    path = "标注.json"
    all_event_information, event_label = get_event_json(path)
    data_informations = [
        data_information(information) 
        for information in all_event_information
    ]
    # 获取"1"事件
    yes_event = []
    pool = Pool(cpu_count())
    results = [
        pool.apply_async(data_inf.read_dataset) 
        for data_inf in data_informations
    ]
    pool.close()
    pool.join()
    for result in results:
        y_temp = result.get()
        max_row = np.size(y_temp, axis=0)
        for i in range(0, np.size(y_temp, axis=1)):
            for j in range(0, max_row, 500):
                if j + 500 > max_row:
                    break
                yes_event.append(
                    np.array(myfft(y_temp[j : j + 500, i]))
                    .reshape(250, 1)
                )
    # 获取"0"事件            
    no_event = []
    for i in range(28):
        path = "/home/T02053124/数据/no_event/" + str(i) + ".csv"
        temp = np.loadtxt(path, delimiter=",")
        for i in range(np.size(temp, axis=1)):
            no_event.append(np.array(myfft(temp[:500, i]))
                            .reshape(250, 1))
    
    # 设置数据集和对应标签
    yes_event = np.array(yes_event)
    no_event = np.array(no_event)
    X = np.append(yes_event, no_event, axis=0)
    Y = np.append(np.ones(yes_event.shape[0]), 
                  np.zeros(no_event.shape[0]))
    return X, Y


def get_XY(step_number: int = 16):  # step_number必须小于32，最好为2的倍数
    path = "标注.json"
    all_event_information, event_label = get_event_json(path)
    data_informations = [
        data_information(information) 
        for information in all_event_information
    ]
    pool = Pool(cpu_count())
    results = [
        pool.apply_async(data_inf.read_dataset) 
        for data_inf in data_informations
    ]
    pool.close()
    pool.join()
    for i in range(len(data_informations)):
        data_informations[i].dataset = results[i].get()
    X, Y = [], []
    for data_inf in data_informations:
        datasets = get_split_data(data_inf.dataset, 
                                  step_number)
        labels = [x for x in data_inf.data_label 
                  for i in range(len(datasets))]
        for block in datasets:
            X.append(block)
        for label in labels:
            Y.append(label)
    return np.array(X), np.array(Y), event_label


def train_and_test_data(X, Y, event_label):
    # 由于X中数据按事件排列有序，因此利用循环选择每个事件的80%作为测试集，最终拼接得出最终的测试集和训练集
    X_train, X_test, Y_train, Y_test = [], [], [], []
    i = 0
    for event in event_label:
        X_temp, Y_temp = [], []
        while Y[i] == event_label[event]:
            X_temp.append(X[i])
            Y_temp.append(Y[i])
            i += 1
            if i == np.size(Y):
                break
        X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(
            X_temp, Y_temp, test_size=0.2, random_state=16
        )
        for j in X_train_temp:
            X_train.append(j)
        for j in X_test_temp:
            X_test.append(j)
        Y_train.extend(Y_train_temp)
        Y_test.extend(Y_test_temp)
    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)
