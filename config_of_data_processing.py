#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/18 15:11
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : config_of_training.py
# @Software: PyCharm Community Edition

window_size = 300   # *0.1ms
window_step = 10     # *0.1ms
signal_max_len = 18000 + window_size
# 部分信号最大值在1500ms之后，此行用于确认
# signal_max_len = 23000 + window_size

signal_time = int(signal_max_len / 10)  # *1ms

# 接下来的 smooth_window_size 窗口内大于 H_start_th 含ELM认为ELMy H模开始
# 接下来的 smooth_window_size 窗口内小于 H_end_th   含ELM认为ELMy H模结束
# 两段H模小于gap间隔认为是一段H模
smooth_window_size = 20
H_start_th = 16
H_end_th = 10
gap = 50    # *1ms

Signal_th = 0.4
default_Freq = 10  # 1/ms:处理数据时默认的频率

# 按照年份划分的放电时期对应的炮号区间
interval_of_shots = [[10595, 13433], [13434, 15117], [15118, 18218], [18219, 21325], [21326, 23073],
                     [23074, 26578], [26579, 28051], [28052, 29892], [29893, 31982], [31983, 35915]]

# error_recognition = [13223, 13823, 15417, 15843, 17787, 17792, 18066, 18067, 18140, 19124,
#                      19695, 21635, 21640, 21725, 22262, 22463, 22495, 24196, 24368, 25847,
#                      27252, 27253, 28853, 28908, 30890, 30949, 31311, 34156, 34157, 34295,
#                      34303, 34337, 34453, 34501]
