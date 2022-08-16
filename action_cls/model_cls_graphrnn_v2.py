"""
    Compared with model_baseline, do not use correlation output for skip link
    Compared to model_baseline_fixed, added return values to test whether nsample is set reasonably.
"""

import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))

#import tf_util
from net_utils import *

# Import graph-rnn and pointnet 
from pointnet2_color_feat_states import *
from graphrnn import *


def placeholder_inputs(batch_size, num_point, num_frames):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * num_frames, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, num_frames, is_training, bn_decay=None):

    
    print('\n --- GRAPH-RNN For Action Classification --- \n')
    
    """ Input:
            point_cloud: [batch_size, num_point * num_frames, 3]
        Output:
            net: [batch_size, num_class] 
    """
    
    
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // num_frames

    l0_xyz = point_cloud
    l0_time = tf.concat([tf.ones([batch_size, num_point, 1]) * i for i in range(num_frames)], \
            axis=-2)
    l0_points = tf.concat([point_cloud[:, :, 3:], l0_time], axis=-1)
    
    num_samples = 8
    knn = True
    activation = None
    
    sampled_points_down1 = num_point/2 #Not used
    sampled_points_down2 = num_point/2/2
    sampled_points_down3 = num_point/2/2/2


    
    cell_feat_1 = GraphFeatureCell(radius=1.0+1e-6, nsample=2*num_samples, out_channels=64, knn=knn, pooling='max', activation =activation)
    cell_feat_2 = GraphFeatureCell(radius=1.0+1e-8, nsample=2*num_samples, out_channels=128, knn=knn, pooling='max', activation =activation)
    cell_feat_3 = GraphFeatureCell(radius=1.0+1e-12, nsample=1*num_samples, out_channels=128, knn=knn, pooling='max', activation =activation)
        
    
    graph_cell1 = GraphRNNCell(radius= 0.1, nsample=num_samples, out_channels=128, knn= knn, pooling='max', activation=activation)
    graph_cell2 = GraphRNNCell(radius= 0.1, nsample=num_samples, out_channels=128, knn= knn, pooling='max', activation=activation)
    graph_cell3 = GraphRNNCell(radius= 0.1, nsample=num_samples, out_channels=128, knn= knn, pooling='max', activation =activation)
    
    
    #STATES
    global_state1 = None
    global_state2 = None
    global_state3 = None
    
    frames = tf.reshape(point_cloud, (batch_size,num_frames,num_point, 3) )
    frames = tf.split(value=frames, num_or_size_splits=num_frames, axis=1)        
    frames = [tf.squeeze(input=frame, axis=[1]) for frame in frames]   
    print("frames", frames)
    
    print(" ========= CONTEXT  ============")
    for i in range(int(num_frames) ):
    	print("frame:", i, "\n")
    	
    	xyz0 = frames[i]
    	color0 = xyz0
    	
    	print("xyz0", xyz0)
    	
    	#print("\n === Downsample Module 1  ====") 
    	xyz1, color1, feat1, states1, _, _ = sample_and_group_with_states(int(sampled_points_down1), radius=1.0+1e-8, nsample= 1, xyz=xyz0,  color=color0, features=None, states = None, knn=True, use_xyz=False) 
    	
    	#xyz1 = xyz0
    	print("xyz1", xyz1)
    	
    	print("\n === CELL 1  Graph-Features ====")
    	with tf.variable_scope('gfeat_1', reuse=tf.AUTO_REUSE) as scope:
	    	out_1 = cell_feat_1((xyz1, None, None, None))
	    	f_xyz1, f_color1, f_feat1, f_states1 = out_1
	    	print("f_xyz1",f_xyz1)
	    	print("f_feat1",f_feat1)
	    	print("f_color1",f_color1)
	    	print("f_states1",f_states1)
	    	print("\n")

    	print("\n === CELL 2  Graph-Features ====")
    	with tf.variable_scope('gfeat_2', reuse=tf.AUTO_REUSE) as scope:
	    	out_2 = cell_feat_2((f_xyz1, None, f_feat1, None))
	    	f_xyz2, f_color2, f_feat2, f_states2 = out_2
	    	print("f_xyz2",f_xyz2)
	    	print("f_feat2",f_feat2)
	    	print("f_color2",f_color2)
	    	print("f_states2",f_states2)
	    	print("\n")	    	
    	
    	
    	print("\n === CELL 3  Graph-Features ====")
    	with tf.variable_scope('gfeat_3', reuse=tf.AUTO_REUSE) as scope:
	    	out_3 = cell_feat_3((f_xyz2, None, f_feat2, None))
	    	f_xyz3, f_color3, f_feat3, f_states3 = out_3
	    	print("f_xyz3",f_xyz3)
	    	print("f_feat3",f_feat3)
	    	print("f_color3",f_color3)
	    	print("f_states3",f_states3)
	    	print("\n")

    	time = tf.fill( (f_xyz3.shape[0],f_xyz3.shape[1],1), (i/1.0))
    	
    	print("\n === CELL 1 GraphRNN ====")
    	with tf.variable_scope('graphrnn_1', reuse=tf.AUTO_REUSE) as scope:
    		global_state1 = graph_cell1( (f_xyz3, None, f_feat3, None, time), global_state1)
    		s_xyz1, s_color1, s_feat1, s_states1, time, extra  = global_state1
    		print("s_xyz1",s_xyz1)
    		print("s_feat1",s_feat1)
    		print("s_color1",s_color1)
    		print("s_states1",s_states1)
    		print("\n") 

    	print("\n === CELL 2 GraphRNN ====")
    	xyz2, color2, feat2, states2, _, _ = sample_and_group_with_states(int(sampled_points_down2), radius=1.0+1e-20, nsample= 4 , xyz=s_xyz1,  color=s_xyz1, features=s_feat1, states = s_states1, knn=True, use_xyz=False)
    	feat2 = tf.reduce_max(feat2, axis=[2], keepdims=False, name='maxpool')
    	states2 = tf.reduce_max(states2, axis=[2], keepdims=False, name='maxpool')
    	time = tf.fill( (xyz2.shape[0],xyz2.shape[1],1), (i/1.0))
    	with tf.variable_scope('graphrnn_2', reuse=tf.AUTO_REUSE) as scope:
    		global_state2 = graph_cell2( (xyz2, None, feat2, states2, time), global_state2)
    		s_xyz2, s_color2, s_feat2, s_states2, time, extra = global_state2
    		print("s_xyz2",s_xyz2)
    		print("s_feat2",s_feat2)
    		print("s_color2",s_color2)
    		print("s_states2",s_states2)
    		print("\n")     	 

    	print("\n === CELL 3 GraphRNN ====")
    	xyz3, color3, feat3, states3, _, _ = sample_and_group_with_states(int(sampled_points_down3), radius=1.0+1e-20, nsample= 4 , xyz=s_xyz2,  color=s_xyz2, features=s_feat2, states = s_states2, knn=True, use_xyz=False)
    	feat3 = tf.reduce_max(feat3, axis=[2], keepdims=False, name='maxpool')
    	states3 = tf.reduce_max(states3, axis=[2], keepdims=False, name='maxpool')
    	time = tf.fill( (xyz3.shape[0],xyz3.shape[1],1), (i/1.0))
    	with tf.variable_scope('graphrnn_3', reuse=tf.AUTO_REUSE) as scope:
    		global_state3 = graph_cell3( (xyz3, None, feat3, states3, time), global_state3)
    		s_xyz3, s_color3,s_feat3, s_states3, time, extra = global_state3
    		print("s_xyz3",s_xyz3)
    		print("s_feat3",s_feat3)
    		print("s_color3",s_color3)
    		print("s_states3",s_states3)
    		print("\n")     
    		    	
    
    print("\n === Fully Connected Layers GraphRNN ====\n") 
    # Use only the final states 3 
    print("s_states3",s_states3)
    with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
    	net = tf.layers.conv1d(inputs=s_states3, filters=128, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='fc1')
    	print("net", net)
    	net = tf.layers.conv1d(inputs=net, filters=20, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc2')
    net = tf.reduce_max(net, axis=[1], name='maxpool')

    print("\n")
    print("net", net)
    print('end_points', end_points)
    
    return net, end_points




def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024*2,6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
