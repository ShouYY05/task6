import tensorflow as tf
import numpy as np
IMAGE_SIZE=32#每张图片分辨率
NUM_CHANNELS=3#输入图片通道数
vCONV_SIZE=3
vCONV_KERNEL_NUM=[None,16,32,64,128,128]
vFC_SIZE_1=100
vFC_SIZE_2=40
def get_weight(shape,regularizer):#带有正则化
    w=tf.Variable(tf.random_normal(shape=shape,stddev=0.1),dtype=tf.float32)
    if regularizer!=None :
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
def get_bias(shape):
    b=tf.Variable(tf.zeros(shape=shape))
    return b
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")
  
def max_pool_2x2_pad0(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#前向传播
def con_relu(x,index_ceng=int,regularizer=float,conv_time=int):
    conv=x
    for i in range(conv_time):
        shape_x=int(np.shape(conv)[-1])
        conv_w = get_weight(shape=(vCONV_SIZE, vCONV_SIZE,shape_x, vCONV_KERNEL_NUM[index_ceng]), regularizer=regularizer)
        conv_b = get_bias(shape=vCONV_KERNEL_NUM[index_ceng])
        conv=conv2d(conv,conv_w)
        conv=tf.nn.relu(tf.nn.bias_add(conv,conv_b))
    return conv
def forward(x,train,regularizer):
    conv1=con_relu(x,index_ceng=1,regularizer=regularizer,conv_time=1)
    pool1=max_pool_2x2_pad0(conv1)

    conv2=con_relu(pool1,index_ceng=2,regularizer=regularizer,conv_time=1)
    pool2=max_pool_2x2_pad0(conv2)

    conv3=con_relu(pool2,index_ceng=3,regularizer=regularizer,conv_time=2)
    pool3 = max_pool_2x2_pad0(conv3)

    conv4 = con_relu(pool3, index_ceng=4, regularizer=regularizer,conv_time=2)
    pool4 = max_pool_2x2_pad0(conv4)

    conv5 = con_relu(pool4, index_ceng=5, regularizer=regularizer,conv_time=2)
    pool5 = max_pool_2x2_pad0(conv5)

    pool_shape = pool5.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]  # 从 list 中依次取出矩阵的长宽及深度，并求三者的乘积，得到矩阵被拉长后的长度
    reshaped_x = tf.reshape(pool5, [pool_shape[0], nodes])  # 将 pool2 转换为一个 batch 的向量再传入后续的全连接

    fc1_w = get_weight([nodes, vFC_SIZE_1], regularizer)
    fc1_b = get_bias([vFC_SIZE_1])
    fc1 = tf.nn.relu(tf.matmul(reshaped_x, fc1_w) + fc1_b)
    if train: fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([vFC_SIZE_1, vFC_SIZE_2], regularizer)
    fc2_b = get_bias([vFC_SIZE_2])
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
    if train: fc2 = tf.nn.dropout(fc2, 0.5)

    fc3_w = get_weight([vFC_SIZE_2, OUTPUT_NOOD], regularizer)
    fc3_b = get_bias(OUTPUT_NOOD)
    y = tf.matmul(fc2, fc3_w) + fc3_b
    return y
