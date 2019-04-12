#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/11 9:29
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : api_of_network.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import pickle
import numpy as np

# ============================ 创建网络 ============================
def create_conv_model(fingerprint_input, model_settings, is_training, DropOutProb=0.5):
    """Builds a standard convolutional model.
          (fingerprint_input)
              v
          [Conv1D]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
            [Relu]
              v
          [MaxPool]
              v
          [Conv1D]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
            [Relu]
              v
          [MaxPool]
              v
          [Conv1D]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
            [Relu]
              v
          [MaxPool]
              v
          [Conv1D]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
            [Relu]
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
           (Output)
    """

    """=============================模型超参数==============================="""
    model_hyperpars = {
        'first_filter_width': 50,
        'first_filter_count': 16,
        'second_filter_width': 50,
        'second_filter_count': 16,
        'third_filter_width': 50,
        'third_filter_count': 12,
        'fourth_filter_width': 50,
        'fourth_filter_count': 8,
        'second_fc_element_count': 25
    }

    """============================reshape输入=============================="""
    input_time_size = model_settings['input_time_size']
    fingerprint_3d = tf.reshape(fingerprint_input, [-1, input_time_size, 1])

    """============================记录variable=============================="""
    variable = []

    """=============================第一卷积层==============================="""
    first_filter_width = model_hyperpars['first_filter_width']
    first_filter_count = model_hyperpars['first_filter_count']

    with tf.name_scope('Conv_1'):
        first_weights = tf.Variable(
            tf.truncated_normal([first_filter_width, 1, first_filter_count],
                                stddev=0.01), name="weight")
        first_bias = tf.Variable(tf.zeros([first_filter_count]), name="bias")
        variable.append(first_weights)
        variable.append(first_bias)

        first_conv = tf.add(tf.nn.conv1d(fingerprint_3d, first_weights, 1, 'SAME'), first_bias, name="conv1d")
        first_relu = tf.nn.relu(first_conv, name="relu")
        first_dropout = tf.nn.dropout(first_relu, DropOutProb, name="dropout") if is_training else first_relu
        first_max_pool = tf.nn.pool(input=first_dropout, window_shape=[2], strides=[2],
                                    pooling_type="MAX", padding='SAME', name="pool")

    """=============================第二卷积层==============================="""
    second_filter_width = model_hyperpars['second_filter_width']
    second_filter_count = model_hyperpars['second_filter_count']

    with tf.name_scope('Conv_2'):
        second_weights = tf.Variable(
            tf.truncated_normal([second_filter_width, first_filter_count, second_filter_count],
                                stddev=0.01), name="weight")
        second_bias = tf.Variable(tf.zeros([second_filter_count]), name="bias")
        variable.append(second_weights)
        variable.append(second_bias)

        second_conv = tf.add(tf.nn.conv1d(first_max_pool, second_weights, 1, 'SAME'), second_bias, name="conv1d")
        second_relu = tf.nn.relu(second_conv, name="relu")
        second_dropout = tf.nn.dropout(second_relu, DropOutProb, name="dropout") if is_training else second_relu
        second_max_pool = tf.nn.pool(input=second_dropout, window_shape=[2], strides=[2],
                                     pooling_type="MAX", padding='SAME', name="pool")

    """=============================第三卷积层==============================="""
    third_filter_width = model_hyperpars['third_filter_width']
    third_filter_count = model_hyperpars['third_filter_count']

    with tf.name_scope('Conv_3'):
        third_weights = tf.Variable(
            tf.truncated_normal([third_filter_width, second_filter_count, third_filter_count],
                                stddev=0.01), name="weight")
        third_bias = tf.Variable(tf.zeros([third_filter_count]), name="bias")
        variable.append(third_weights)
        variable.append(third_bias)

        third_conv = tf.add(tf.nn.conv1d(second_max_pool, third_weights, 1, 'SAME'), third_bias, name="conv1d")
        third_relu = tf.nn.relu(third_conv, name="relu")
        third_dropout = tf.nn.dropout(third_relu, DropOutProb, name="dropout") if is_training else third_relu
        third_max_pool = tf.nn.pool(input=third_dropout, window_shape=[2], strides=[2],
                                    pooling_type="MAX", padding='SAME', name="pool")

    """=============================第四卷积层==============================="""
    fourth_filter_width = model_hyperpars['fourth_filter_width']
    fourth_filter_count = model_hyperpars['fourth_filter_count']

    with tf.name_scope('Conv_4'):
        fourth_weights = tf.Variable(
            tf.truncated_normal([fourth_filter_width, third_filter_count, fourth_filter_count],
                                stddev=0.01), name="weight")
        fourth_bias = tf.Variable(tf.zeros([fourth_filter_count]), name="bias")
        variable.append(fourth_weights)
        variable.append(fourth_bias)

        fourth_conv = tf.add(tf.nn.conv1d(third_max_pool, fourth_weights, 1, 'SAME'), fourth_bias, name="conv1d")
        fourth_relu = tf.nn.relu(fourth_conv, name="relu")
        fourth_dropout = tf.nn.dropout(fourth_relu, DropOutProb, name="dropout") if is_training else fourth_relu

        fourth_conv_shape = fourth_dropout.get_shape()
        fourth_conv_output_width = fourth_conv_shape[1]
        fourth_conv_element_count = int(fourth_conv_output_width * fourth_filter_count)
        flattened_fourth_conv = tf.reshape(fourth_dropout,
                                           [-1, fourth_conv_element_count])

    """============================第一全连接层=============================="""
    second_fc_element_count = model_hyperpars['second_fc_element_count']
    label_count = model_settings['label_count']

    with tf.name_scope('Fc_1'):
        first_fc_weights = tf.Variable(
            tf.truncated_normal([fourth_conv_element_count, second_fc_element_count],
                                stddev=0.01), name="weight")
        first_fc_bias = tf.Variable(tf.zeros([second_fc_element_count]), name="bias")
        variable.append(first_fc_weights)
        variable.append(first_fc_bias)

        first_fc = tf.add(tf.matmul(flattened_fourth_conv, first_fc_weights), first_fc_bias)

    """============================第二全连接层=============================="""
    with tf.name_scope('Fc_2'):
        second_fc_weights = tf.Variable(
            tf.truncated_normal([second_fc_element_count, label_count],
                                stddev=0.01), name="weight")
        second_fc_bias = tf.Variable(tf.zeros([label_count]), name="bias")
        variable.append(second_fc_weights)
        variable.append(second_fc_bias)

        second_fc = tf.add(tf.matmul(first_fc, second_fc_weights), second_fc_bias, name="output")

    # 返回variable 以便裁剪
    return variable, second_fc


# ============================pickle api============================
# 这部分是创建和获取pickle文件的api
class pickle_reader(object):
    def __init__(self, pickles_dir):
        self.pickles_dir = pickles_dir
        [self.train_slices, self.train_labels] = self.get_pickle(pickles_dir + '/TrainData.pickle')
        [self.val_slices, self.val_labels] = self.get_pickle(pickles_dir + '/ValData.pickle')
        [self.test_slices, self.test_labels] = self.get_pickle(pickles_dir + '/TestData.pickle')
        self.slice_length = self.train_slices.shape[1]

    @staticmethod
    def get_pickle(pickle_path):
        with open(pickle_path, 'rb') as f:
            signals = pickle.load(f)
            return signals['Slices'], signals['Labels']

    def get_batch(self, batch_size, mode):
        assert mode in ['Train', 'Val', 'Test']
        mode = mode.lower()
        set_size = eval("self."+mode+"_slices.shape[0]")
        random_idx_set = np.random.permutation(set_size)[:batch_size]
        return eval("self."+mode+"_slices[random_idx_set], self."+mode+"_labels[random_idx_set]")


# ============================调用模型============================
# 这部分是已经获得训练的model之后后续处理的api
def init_NN(pb_file_path=r"./model/frozen_model_1.pb"):
    """
    从 pb_file_path 读取pb文件，并返回sess及输入、输出节点
    :param pb_file_path: 
    :return: 
    """
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

    with open(pb_file_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    finger_input = sess.graph.get_tensor_by_name("input:0")
    output = sess.graph.get_tensor_by_name("output:0")
    return sess, finger_input, output


def freeze_graph(input_meta, output_graph):
    '''
    :param input_meta:    需要冻结的.meta文件
    :param output_graph:        PB模型保存路径
    :return:
    example：
        iinput_meta='2018101803/checkpoint/conv.ckpt-40000'
        out_pb_path="model/example_model.pb"
        freeze_graph(input_meta, out_pb_path)
    '''
    import tensorflow as tf
    # checkpoint = tf.train.get_checkpoint_state(model_folder)  #检查目录下meta文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path       #得ckpt文件路径

    # 打印网络中的节点
    # from pprint import pprint
    # for op in graph.get_operations():
    #     pprint(op.name, op.values())

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "output,input"
    saver = tf.train.import_meta_graph(input_meta + '.meta', clear_devices=True)
    graph = tf.get_default_graph()          # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_meta)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出


def create_tensorboard_logs(model, save_path="log/"):
    import tensorflow as tf
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    if model.endswith(".pb"):
        graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
        tf.import_graph_def(graph_def, name='graph')

    elif model.endswith(".meta"):
        tf.train.import_meta_graph(model)

    tf.summary.FileWriter(save_path, graph)
    # 在cmd中输入以下命令(可能需要绝对路径) 查看tensorboard
    # tensorboard --logdir  ".\my_H_mode\log\"