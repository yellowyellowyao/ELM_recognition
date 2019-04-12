#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/26 9:36
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : api_of_data_processing.py
# @Software: PyCharm Community Edition

import scipy.io as sio
import os
import numpy as np
import pymysql
from my_hdf import my_hdf5 as mh
from config_of_data_processing import *
import matplotlib.pyplot as plt
np.seterr(divide='ignore',invalid='ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# ============================data processing============================
# 这部分是已经获得训练的model之后后续处理的api

def get_label_from_mat(path=r".\HModeDataSet"):
    """
    :param path: mat文件所在文件夹
    :return:     含H模信号的炮号
    """
    ishmode = []
    for (path, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            # print(filename)
            filepath = path + "/" + filename
            MatContent = sio.loadmat(filepath)
            if MatContent['bHMode'] == 1:
                ishmode.append(MatContent['ShotNum'].reshape(-1))
    return ishmode


def get_recog_H_mode(start_shot=10595, end_shot=37081, isprint=True):
    """
    :param start_shot: 开始炮号
    :param end_shot:  结束炮号
    :param isprint:  是否打印信息
    :return:  开始与结束炮号之间识别出的H模
    """

    start_shot = max(10595, start_shot)
    end_shot = min(37081, end_shot)
    assert end_shot >= start_shot
    shots = np.load("./recog_result/Hmode_shots of 10595-35914.npy")
    th1 = np.where(shots >= start_shot)[0][0]
    th2 = np.where(shots <= end_shot)[0][-1]
    output = shots[th1:th2]
    if isprint:
        print("Nearest shots is {:},{:}\nAnd,shots number is {:}".format(shots[th1],shots[th2],len(output)))
    return output


def append_shots_to_recog_H_mode(add_shots):
    shots = np.load("./recog_result/Hmode_shots of 10595-35914.npy")
    shots = np.append(shots, add_shots)
    shots = np.array(list(set(shots)))
    shots = np.sort(shots)
    np.save("./recog_result/Hmode_shots of 10595-35914.npy",shots)
    print("Append shots {}: successfully".format(add_shots))


def del_shots_of_recog_H_mode(del_shots, password=""):
    if password is not "yaoh":
        print("You should be very careful about this function!!!")
        print("You should be very careful about this function!!!")
        print("You should be very careful about this function!!!")
        raise KeyError("You should be very careful about this function!!!")

    shots = list(np.load("./recog_result/Hmode_shots of 10595-35914.npy"))
    for i in del_shots:
        if i in shots:
            shots.remove(i)
            print("Delete shot {}: successfully".format(i))
        else:
            print("Shot {} not in file".format(i))
    shots = np.array(shots)
    np.save("./recog_result/Hmode_shots of 10595-35914.npy", shots)


def get_hdf5_file(shotnum,isprint=True):
    """
    从HDF5文件读取 HMode_Ha 信号
    :param shotnum:     炮号
    :param isprint:     打印错误信息
    :return:            HMode_Ha 信号
    
    """
    try:
        _, signal = mh.read_channel(shotnum, channel="HMode_Ha")
        if max(signal) <= Signal_th:
            if isprint:
                print("{} Maximum < Signal_th ({}), pass".format(shotnum,Signal_th))
            return None
        freq = mh.get_attrs("T_Freq", shot_number=shotnum, channel="HMode_Ha") / 1000
        if not freq == default_Freq:
            if isprint:
                print("Frequency should be 10/ms,but it is : {}/ms, pass".format(freq))
            return None
        # 部分信号长度不足，进行扩充
        signal = np.append(signal, [0] * signal_max_len)
        return signal
    except:
        if isprint:
            print("Something is wrong,pass")
        return None


def slice_one_shot_and_predict(signal,sess,output,input):
    """
    将信号切片并进行识别
    :param signal:   从 get_hdf5_file 获得
    :param sess:     从 init_NN 获得
    :param output:   从 init_NN 获得
    :param input:    从 init_NN 获得
    :return:         预测结果：一维矩阵
    """
    # 最好先运行一遍sess.run再执行这一模块,因为第一次运行sess.run比较花时间,会影响实时精度
    predicts = np.array([])
    for i in range(0, signal_max_len, window_step):
        fingerprint_input = signal[i:i + window_size].reshape(1, window_size)

        # ELM信号需要大于Signal_th才会出现
        if np.amax(fingerprint_input) < Signal_th:
            predicts = np.append(predicts, 0)
            continue

        # 此处执行归一化
        fingerprint_input = fingerprint_input - np.mean(fingerprint_input)
        fingerprint_input = fingerprint_input / np.std(fingerprint_input)
        # fingerprint_input = (fingerprint_input - 0.9*np.mean(fingerprint_input)) / (
        # np.mean(fingerprint_input) * np.std(fingerprint_input))

        predict = sess.run(output, feed_dict={input: fingerprint_input, })
        predicts = np.append(predicts, int(predict))
    return predicts


def cal_start_and_end_moment(predicts):
    H_mode_moment = []
    for i in range(smooth_window_size, len(predicts)):
        # 判断H模开始节点
        if len(H_mode_moment) % 2 == 0:
            if sum(predicts[i - smooth_window_size:i]) >= H_start_th:
                # print(predicts[i - smooth_window_size:i])
                H_mode_moment.append(i)

                # 两段H模小于此间隔认为是一段H模
                if len(H_mode_moment) > 2 and H_mode_moment[-1] - H_mode_moment[-2] <= gap * default_Freq / window_step:
                    H_mode_moment.pop()
                    H_mode_moment.pop()
                continue

        # 判断H模结束节点
        if len(H_mode_moment) % 2 == 1:
            if i - smooth_window_size < H_mode_moment[-1]:
                continue
            if sum(predicts[i - smooth_window_size:i]) < H_end_th:
                H_mode_moment.append(i)

    # 没有识别出结束节点将信号最后节点认为是结束节点
    if len(H_mode_moment) % 2 == 1:
        H_mode_moment.append(signal_time)

    # 转为np数组方便后续处理
    H_mode_moment = np.array(H_mode_moment).astype(int)
    if len(H_mode_moment) is not 0:
        print("H模区间为：{}".format(H_mode_moment * window_step / default_Freq))
    return H_mode_moment


def cal_H_mode_interval(H_mode_moment):
    interval = np.array([0.] * int(signal_max_len / window_step))
    for i in range(0, len(H_mode_moment), 2):
        interval[H_mode_moment[i]:H_mode_moment[i + 1]] = [1]
    return interval


def save_recog_result_to_npy(hmode_shots,shots):
    hmode_shots = hmode_shots.astype(int)
    # print(hmode_shots)
    print("In {}-{},number of H_mode shots is {} ".format(shots[0],shots[-1],len(hmode_shots)))
    np.save("./recog_result/Hmode_shots of " + str(shots[0]) + "-" + str(shots[-1]) + ".npy", hmode_shots)
    # for linux
    # np.save("/home/huangyao/code/my_H_mode/recog_result/Hmode_shots of " +
    #         str(shots[0]) + "-" + str(shots[-1]) + ".npy", hmode_shots)


def plot_recog_result(signal, shotnum, predicts=None,
                      interval=None, mode=0, maxshow=False, language="English"):
    """
    
    :param signal:      原始信号
    :param shotnum:     炮号
    :param predicts:    单个时间片预测结果
    :param interval:    H模区间
    :param mode:    
        0：画出所有结果
        1：仅画出原始信号和H模区间
        2: 仅画出原始信号和单个时间片
        3：仅画出原始信号
    :param maxshow:     是否全屏显示
    :param language:    选择坐标轴语言，"English"为英文 ,其他输入皆为中文 
    :return: None
    """
    def only_plot_signal():
        if language is "English":
            plt.title("Recognition of ELM \n shot number：{:0}".format(shotnum),size=30)
            plt.xlabel("Time(ms)", size=26)
        else:
            plt.title("ELM切片识别结果\n 炮号：{:0}".format(shotnum),size=30)
            plt.xlabel("时间(ms)", size=26)
        # plt.ylabel("I_Div_Ha(A.U.)", size=26)
        plt.ylabel("Ha(A.U.)", size=26)
        plt.xticks(np.arange(0, signal_time, 100), size=20)
        plt.yticks(size=20)
        # plt.plot(np.arange(0, signal_time, 0.1), signal[0:signal_max_len], linewidth=0.5, label='I_Div_Ha')
        plt.plot(np.arange(0, signal_time, 0.1), signal[0:signal_max_len], linewidth=0.5, label='Ha')
        plt.legend(loc='upper right', prop={"size": 20})

    def add_interval():
        if language is "English":
            plt.plot(np.arange(0, signal_time, window_step / default_Freq), interval, label='Interval of ELM')
        else:
            plt.plot(np.arange(0, signal_time, window_step / default_Freq), interval, label='ELM区间')

    def plot_slice_result():
        if language is "English":
            plt.xlabel("Time(ms)", size=26)
            plt.ylabel(" Recognition result ", size=26)
            plt.plot(np.arange(0, signal_time, window_step / default_Freq),
                     predicts, linewidth=1, label='Result of single slice')
        else:
            plt.xlabel("时间(ms)", size=26)
            plt.ylabel("识别结果", size=26)
            plt.plot(np.arange(0, signal_time, window_step / default_Freq),
                     predicts, linewidth=1, label='单个时间点H模识别结果')
        plt.xticks(np.arange(0, signal_time, 100), size=20)
        plt.yticks(size=20)
        plt.legend(loc='upper right',prop={"size":20})

    if mode is 0:
        plt.subplot(2, 1, 1)
        only_plot_signal()
        add_interval()
        plt.legend(loc='upper right', prop={"size": 12})

        plt.subplot(2, 1, 2)
        plot_slice_result()
    elif mode is 1:
        only_plot_signal()
        add_interval()
        plt.legend(loc='upper right', prop={"size": 20})
    elif mode is 2:
        plt.subplot(2, 1, 1)
        only_plot_signal()
        plt.subplot(2, 1, 2)
        plot_slice_result()
    elif mode is 3:
        only_plot_signal()
        plt.legend(loc='upper right', prop={"size": 20})

    if maxshow:
        plt.get_current_fig_manager().window.showMaximized()

    plt.show()


def plot_recog_time_cost(file,languge="English",*args):
    time_cost = np.loadtxt(file)
    for file_dir in args:
        time_cost = np.append(time_cost,np.loadtxt(file_dir))
    a = np.sort(time_cost)
    del time_cost

    # TODO:想想怎么自动画区间，额，不想想了，就这样吧,手动调节就好了
    # group_distance = (max(a)-min(a))//0.5
    group_distance = 0.005
    bins = np.arange(0.35, 0.85, group_distance)
    n, _, _ = plt.hist(a, bins=bins, normed=True, color='steelblue', alpha=1)

    plt.xticks(bins, size=12)
    plt.xticks(np.arange(0.0, 0.85, 0.05), size=16)
    plt.yticks(size=16)
    plt.grid(True)

    if languge is "English":
        plt.title(" Recognition time of H-mode \nfrequency distribution histogram", size=24)
        plt.xlabel("Recognition time of single slice (ms)", size=22)
        plt.ylabel("Frequency/Group interval({:}ms)".format(group_distance), size=22)
        plt.text(0.8 * max(a), 0.8 * max(n), "\n\nTotal slices ： {:}\n\n"
                                             "minimum time ： {:.2f} ms\n\n"
                                             "maximum time ： {:.2f} ms\n\n"
                                             "Average time ： {:.2f} ms " .format(len(a),a[0],a[-1],np.mean(a)),size=20)
    else:
        plt.title(" H模切片识别时间 \n频率分布直方图", size=18)
        plt.xlabel("一个时间点（300个采样点）的识别时间 (单位: ms)", size=16)
        plt.ylabel("频率/组距({:}ms)".format(group_distance), size=16)
        plt.text(0.8 * max(a), 0.8 * max(n), "总时间点数量： {:}\n\n"
                                             "最小识别时间： {:.2f} ms\n\n"
                                             "最大识别时间： {:.2f} ms\n\n"
                                             "平均识别时间： {:.2f} ms ".format(len(a), a[0], a[-1], np.mean(a)), size=14)
    plt.show()


# ============================mySQL接口============================
def connect_to_mySQL(host='192.168.9.222',user='yhuang',password='123456'):
    database_config = {'host': host,
                       'user': user,
                       'password': password}
    database_name = 'fusion-ai'
    connect = pymysql.Connect(host=database_config['host'],
                              user=database_config['user'],
                              password=database_config['password'],
                              db=database_name,
                              charset='utf8',
                              cursorclass=pymysql.cursors.DictCursor)
    return connect


def write_to_mySQL(connect,H_mode_moment,shotnum):
    for i in range(0, len(H_mode_moment), 2):
        H_mode_start = float(H_mode_moment[i])
        H_mode_end = float(H_mode_moment[i + 1])
        sql = 'INSERT INTO h_mode_shots_hl2a (ShotNum, HModeStart, HModeStop) VALUES (%s, %s, %s)'
        with connect.cursor() as Cursor:
            Cursor.execute(sql, (shotnum, H_mode_start, H_mode_end))
        connect.commit()








