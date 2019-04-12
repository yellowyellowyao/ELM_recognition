#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 9:31
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : check_one_shot.py
# @Software: PyCharm Community Edition
from api_of_data_processing import *
from api_of_network import init_NN

def recog_hmode(shots):
    """
     function: 识别炮号是否为H模并作图
    :param shots:  需要识别的炮号序列
    :return:  None
    """

    # =============================加载pb模型=================================
    sess, finger_input, output = init_NN()
    # 先初始化sess.run一次以加快速度
    _ = sess.run(output, feed_dict={finger_input: np.zeros((1, window_size)), })

    # =============================开始逐炮识别=================================
    for shot_num in shots:
        print("shot_num is : {:}".format(shot_num))

        # =============================读取hdf5文件=================================
        signal = get_hdf5_file(shot_num)
        if signal is None:
            continue

        # ============================单个时间片识别============================
        predicts = slice_one_shot_and_predict(signal, sess, output, finger_input)
        if sum(predicts) < 10:
            print("sum of predicts less than 10,pass")
            continue

        # ============================识别H模开始与结束时间点============================
        H_mode_moment = cal_start_and_end_moment(predicts)
        interval = cal_H_mode_interval(H_mode_moment)*max(signal)

        # ============================作图============================
        plot_recog_result(signal, shot_num, predicts=predicts,
                          interval=interval, mode=0, maxshow=1, language="English")
        # mode:
        # 0：画出所有结果
        # 1：仅画出原始信号和H模区间
        # 2: 仅画出原始信号和单个时间片
        # 3：仅画出原始信号

if __name__ == '__main__':
    # good shots: 20000 20001 22025
    #  bad shots: 22190 22733 22882 24276

    shots =[36771]
    recog_hmode(shots)

