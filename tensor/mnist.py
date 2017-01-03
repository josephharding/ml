
import numpy as np

import struct
import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def get_big_endian_byte(s):
  return struct.unpack(">B", s)[0]

def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # create model
  x = tf.placeholder(tf.float32, [None, (28*28)])

  W = tf.Variable(tf.zeros([(28*28), 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b # the model definition - still just a template though right now

  # define loss
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  init = tf.global_variables_initializer()

  sess = tf.Session()
  sess.run(init)

  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # now our model contains data since it was trained
  guess = tf.argmax(y, 1)
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # so I need to write my own mnist.joe_test that contains a test image and label

  fake_labels = np.ndarray(shape=(10000, 10), dtype=float, order='C')
  for idx in range(10000):
    fake_labels.itemset((idx, 0), 1.)

  #print("test labels", mnist.test.labels)
  #print("fake labels", fake_labels)
  #print("image data", mnist.test.images)
  #for idx in range(10):
  #  print(mnist.test.images[idx])

  # 3274 total bytes
  pixels = []
  eval_image = np.ndarray(shape=(1, 28 * 28), dtype=float, order='C')
  f = open(FLAGS.eval_image, 'rb')
  try:
    f.seek(0)
    b = struct.unpack(">c", f.read(1))
    m = struct.unpack(">c", f.read(1))
    size = struct.unpack(">BBBB", f.read(4))
    print ("header bytes", b, m, size)

    f.seek(10)
    starting = struct.unpack(">BBBB", f.read(4))[0]
    print ("starting address", starting)

    f.seek(14)
    header_bytes = struct.unpack(">BBBB", f.read(4))
    print ("header bytes", header_bytes)
    
    f.seek(18)
    width = struct.unpack(">BBBB", f.read(4))
    height = struct.unpack(">BBBB", f.read(4))
    print ("width, height", width, height)
    
    f.seek(28)
    bits_per_pixel = struct.unpack(">BB", f.read(2))[0]
    print ("bits per pixel", bits_per_pixel)
    
    f.seek(30)
    compression_method = struct.unpack(">BBBB", f.read(4))
    print ("compression method", compression_method)
    
    f.seek(starting)

    n_rows = 28
    n_cols = 28
    for j in range(n_rows):
      row = []
      for i in range(n_cols):
        r = get_big_endian_byte(f.read(1))
        g = get_big_endian_byte(f.read(1))
        b = get_big_endian_byte(f.read(1))
        if bits_per_pixel > 24:
          a = get_big_endian_byte(f.read(1))

        val = (float(r) / 255.)
        row.append(val)
        pixels.append(val)
      
      #print(row)

  finally:
    f.close()
 
  # yes - the array is fed in backwards and I don't correct it - I guess this is because the training data is fed in backwards
  #pixels = pixels[::-1]
  for idx in range(len(pixels)):
    eval_image.itemset(idx, pixels[idx])

  #seven_label = np.ndarray(shape=(1, 10), dtype=float, order='C')
  #seven_label.itemset((0, 7), 1.) # in the one-hot vector set the 7th index to 1
  
  #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels }))
  print(sess.run(guess, feed_dict={x: eval_image}))
  print(sess.run(y, feed_dict={x: eval_image}))


if __name__ == '__main__':


  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--eval_image', type=str, default='../reading_bins/images/image0.bmp',
                      help='Image to test model')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
