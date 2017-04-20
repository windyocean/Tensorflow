import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

def adjust(x, axis):
  shrink_reduce_sum = tf.Variable(tf.reduce_sum(x, axis = axis))#, dtype = tf.float32)
  #print(shrink_reduce_sum.eval()) 
 
  weighted_sum = tf.Variable(tf.zeros([1], dtype = tf.float32), dtype = tf.float32)
  #print(tf.shape(shrink_reduce_sum).eval()[0])
  for i in range(tf.shape(shrink_reduce_sum).eval()[0]):
    weighted_sum = tf.cast(tf.add(weighted_sum, tf.multiply(tf.cast(shrink_reduce_sum[i], tf.float32), tf.constant(i, dtype = tf.float32))), dtype = tf.float32)
  center_original = tf.cast(tf.divide(tf.shape(shrink_reduce_sum), tf.constant(2)), tf.float32)
  
  #print('weighted_sum is', weighted_sum.eval())
  print('tf.shape(shrink_reduce_sum)', tf.shape(shrink_reduce_sum).eval())
  tf.Print(tf.shape(shrink_reduce_sum).eval(), ['this is tf.Print',  tf.shape(shrink_reduce_sum).eval()])
  print('center_original is ', center_original.eval())
  #print('tf.subtract(... is' , tf.subtract(tf.divide(weighted_sum, tf.cast(tf.shape(shrink_reduce_sum),tf.float32)), center_original).eval())
  center_after = tf.cast(tf.subtract(tf.divide(weighted_sum, tf.cast(tf.shape(shrink_reduce_sum), dtype = tf.float32)), center_original), tf.int32)

  affined_x = tf.Variable(tf.zeros(tf.shape(x)))
  #print('center_after is ', center_after.eval())
  if axis == 0:
    for i in range(tf.shape(x).eval()[1]):
      if center_after.eval() > 0:
        affined_x = x[:, 1 : tf.sub(tf.shape(x)[1], center_after)]
      else:
        affined_x = x[:, center_after : tf.shape(x)[1]]

  elif axis == 1:
    for i in range(tf.shape(x).eval()[0]):
      if center_after.eval() > 0: #tf.cond(tf.greater(center_after, tf.constant(0)).eval(): .... )
        affined_x = x[1 : tf.sub(tf.shape(x)[0], center_after), :]
      else:
        affined_x = x[center_after : tf.shape(x)[0], :]
  else:
    print('wrong affine transform has been detected')
  return x



with tf.Session() as sess:
  tf.global_variables_initializer()
  a = tf.constant([[0,0,0,1,0], [0,0,0,0,0]])
  print(sess.run(adjust(a, 0)))


  #result = sess.run(tf.shape([[1,2,3],[4,5,6]]))
  #print("array is [[1,2,3], [4,5,6]] then ")
  #print(result)
