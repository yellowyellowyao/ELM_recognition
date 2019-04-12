#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/29 14:36
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : config_of_training.py
# @Software: PyCharm Community Edition

# InputConf
window_len = 300
tf_records_path = r".\Pickles"
batch_for_one_time = 128
train_batch_size = 128
val_batch_size = 1024
test_batch_size = 10000

is_training = False

# Optimizer:Adam or SGD
optimizer = 'SGD'
# optimizer = "tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)"

# if SGD
epoch_for_learning_rate = [40000]
learning_rate_list = [0.1]
# if Adam
epoch_num = 30000
learning_rate_max = 0.1
learning_rate_min = 0.01
decay_speed = 10000

# model Settings
ModelSettings = {
  'input_time_size': window_len,
  'label_count': 2,
  }

# gpuSettings
gpu_num = 1


if_load_model = False
model_dir = "./model/checkpoint/"

from datetime import datetime
time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = r"./model/tensorboard/train/"+time_stamp
val_log_dir = r"./model/tensorboard/validation/"+time_stamp

# model_dir = r"./model/checkpoint"
# summaries_dir = r"./2018101803/tensorboard"
# model_path = r"./2018101803/checkpoint"
eval_step_interval = 100
save_step_interval = 1000

