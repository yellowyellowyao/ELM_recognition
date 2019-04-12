#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/18 15:09
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : save_to_mySQL.py
# @Software: PyCharm Community Edition


from api_of_data_processing import *
from api_of_network import init_NN
from config_of_data_processing import *


def prepare_NN(shots,pb_file_path):
    """
    :param shots: range 
    :param pb_file_path: str 
    :return: None 
    """
    # =============================加载pb模型=================================
    sess, finger_input, output = init_NN(pb_file_path)
    # 先初始化sess.run一次以加快速度
    _ = sess.run(output, feed_dict={finger_input: np.zeros((1, window_size)), })

    # ===========================连接MySQL数据库===============================
    connect = connect_to_mySQL()

    # =============================开始逐炮识别=================================
    try:
        H_mode_shots = np.array([]).astype(int)

        for shotnum in shots:
            # =============================读取hdf5文件=================================
            signal = get_hdf5_file(shotnum,isprint=False)
            if signal is None:
                continue

            # =============================神经网络识别切片=================================
            predicts = slice_one_shot_and_predict(signal, sess, output, finger_input)

            # ============================识别H模开始与结束时间点============================
            H_mode_moment = cal_start_and_end_moment(predicts)
            if len(H_mode_moment) > 0:
                H_mode_shots = np.append(H_mode_shots, shotnum)
                print("shotnum is {}".format(shotnum))

            # ============================写入mySQL============================

            write_to_mySQL(connect, H_mode_moment, shotnum)

    except:
        print(str(shotnum)+" is error")
        # shots = shots[:shotnum]
    finally:
        sess.close()
        # ============================存储识别出来的炮号============================
        # 第一次运行时使用该部分代码
        # save_recog_result_to_npy(hmode_shots, shots)
        append_shots_to_recog_H_mode(H_mode_shots)
        pass


shots = range(37079,37082)
prepare_NN(shots,pb_file_path=r"./model/frozen_model_1.pb")
# for linux
# prepare_NN(shots,pb_file_path=r"/home/huangyao/code/my_H_mode/model/frozen_model_1.pb")

