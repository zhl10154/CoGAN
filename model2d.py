from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class CoGAN(object):
  def __init__(self, sess, config, sample_num = 64,
         z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.is_crop = config.is_crop
    self.is_grayscale = (config.c_dim == 1)
    self.batch_size = config.batch_size
    self.sample_num = sample_num
    self.input_height = config.input_height
    self.input_width = config.input_width
    self.output_height = config.output_height
    self.output_width = config.output_width
    self.dataset_name = config.dataset
    self.input_fname_pattern = config.input_fname_pattern
    self.checkpoint_dir = config.checkpoint_dir
    self.c_dim = config.c_dim
    self.L1_lambda = config.L1_lambda

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g1_bn0 = batch_norm(name='g_G1_bn0')
    self.g1_bn1 = batch_norm(name='g_G1_bn1')
    self.g1_bn2 = batch_norm(name='g_G1_bn2')
    self.g2_bn0 = batch_norm(name='g_G2_bn0')
    self.g2_bn1 = batch_norm(name='g_G2_bn1')
    self.g2_bn2 = batch_norm(name='g_G2_bn2')
    self.g1_bn3 = batch_norm(name='g_G1_bn3')
    self.g2_bn3 = batch_norm(name='g_G2_bn3')
    
    self.loss_values = []
    self.fileTitle = 'Turns,D_real_loss,D_fake1_loss,D_fake2_loss,G1_loss,G2_loss, resultD,resultD_G1,resultD_G2,D_loss,G_loss,L1_loss'
    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    if self.is_crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.output_height, self.output_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    inputs = self.inputs

    self.z1 = tf.placeholder(
    tf.float32, [None, self.z_dim], name='z1')
    self.z1_sum = histogram_summary("z1", self.z1)
    self.z2 = tf.placeholder(
    tf.float32, [None, self.z_dim], name='z2')
    self.z2_sum = histogram_summary("z2", self.z2)

    self.G1 = self.generator(self.z1, name="G1")
    self.G2 = self.generator(self.z2, name="G2")
    self.D, self.D_logits = self.discriminator(inputs)

    self.sampler1 = self.generator(self.z1, name="G1", isTrain=False)
    self.sampler2 = self.generator(self.z2, name="G2", isTrain=False)
    self.D_G1, self.D_logits_G1 = self.discriminator(self.G1, reuse=True)
    self.D_G2, self.D_logits_G2 = self.discriminator(self.G2, reuse=True)
    
    self.D_mean = tf.reduce_mean(self.D)
    self.D_G1_mean = tf.reduce_mean(self.D_G1)
    self.D_G2_mean = tf.reduce_mean(self.D_G2)

    self.d_sum = histogram_summary("d", self.D)
    self.d__G1_sum = histogram_summary("d_g1", self.D_G1)
    self.d__G2_sum = histogram_summary("d_g2", self.D_G2)
    self.G1_sum = image_summary("G1", self.G1)
    self.G2_sum = image_summary("G2", self.G2)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    # d loss
    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake_G1 = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_G1, tf.zeros_like(self.D_G1)))
    self.d_loss_fake_G2 = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_G2, tf.zeros_like(self.D_G2)))
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum_G1 = scalar_summary("d_loss_fake_g1", self.d_loss_fake_G1)
    self.d_loss_fake_sum_G2 = scalar_summary("d_loss_fake_g2", self.d_loss_fake_G2)
    self.d_loss = self.d_loss_real + self.d_loss_fake_G1 + self.d_loss_fake_G2
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    # lower score get penalty
    self.L1_loss = self.L1_lambda * tf.reduce_mean(tf.square(self.G1 - self.G2))
    
    # g loss
    self.g_loss_G1 = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_G1, tf.ones_like(self.D_G1))) 
    self.g_loss_G2 = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_G2, tf.ones_like(self.D_G2))) 
    self.L1_loss_sum = scalar_summary("g_loss_l1", self.L1_loss)
    self.g_loss_sum_G1 = scalar_summary("g_loss_g1", self.g_loss_G1)
    self.g_loss_sum_G2 = scalar_summary("g_loss_g2", self.g_loss_G2)
    self.g_loss = self.g_loss_G1 + self.g_loss_G2 + self.L1_loss
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    """Train CoGAN"""
    data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
    #np.random.shuffle(data)

    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z1_sum, self.z2_sum, self.d__G1_sum, self.d__G2_sum,
      self.G1_sum, self.G2_sum, self.d_loss_fake_sum_G1, self.d_loss_fake_sum_G2,
      self.g_loss_sum_G1, self.g_loss_sum_G2, self.g_loss_sum, self.L1_loss_sum])
    self.d_sum = merge_summary(
        [self.z1_sum, self.z2_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_fake_sum_G1, 
         self.d_loss_fake_sum_G2, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

#     use fixed input z
    sample_z1 = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    sample_z2 = sample_z1
    
    sample_files = data[0:self.sample_num]
    sample = [get_image(sample_file,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  is_crop=self.is_crop,
                  is_grayscale=self.is_grayscale) for sample_file in sample_files]
    if (self.is_grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in range(config.epoch):
      data = glob(os.path.join(
        "./data", config.dataset, self.input_fname_pattern))
      batch_idxs = min(len(data), config.train_size) // config.batch_size

      for idx in range(0, batch_idxs):
        batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      is_crop=self.is_crop,
                      is_grayscale=self.is_grayscale) for batch_file in batch_files]
        if (self.is_grayscale):
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)

        # share z values
        batch_z1 = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
        batch_z2 = batch_z1

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={self.inputs: batch_images,self.z1: batch_z1,self.z2: batch_z2})
        self.writer.add_summary(summary_str, counter)
        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={self.z1: batch_z1,self.z2: batch_z2})
        self.writer.add_summary(summary_str, counter)
        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={self.z1: batch_z1,self.z2: batch_z2})
        self.writer.add_summary(summary_str, counter)
                  # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={self.z1: batch_z1,self.z2: batch_z2})
        self.writer.add_summary(summary_str, counter)
        
        errD_fake1 = self.d_loss_fake_G1.eval({self.z1: batch_z1, self.z2: batch_z2})
        errD_fake2 = self.d_loss_fake_G2.eval({self.z1: batch_z1, self.z2: batch_z2})
        errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
        errG = self.g_loss.eval({self.z1: batch_z1, self.z2: batch_z2})
        errD_L1 = self.L1_loss.eval({self.z1: batch_z1, self.z2: batch_z2})
        errG_G1 = self.g_loss_G1.eval({self.z1: batch_z1, self.z2: batch_z2})
        errG_G2 = self.g_loss_G2.eval({self.z1: batch_z1, self.z2: batch_z2})
        resultD = self.D_mean.eval({self.inputs: batch_images})
        resultD_G1 = self.D_G1_mean.eval({self.z1: batch_z1, self.z2: batch_z2})
        resultD_G2 = self.D_G2_mean.eval({self.z1: batch_z1, self.z2: batch_z2})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.1f, d_loss: %.4f, g_loss: %.4f, L1_loss: %.4f" \
          % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake1+errD_fake2+errD_real, errG, errD_L1))

        # record key values
        self.loss_values.append([counter, errD_real, errD_fake1, errD_fake2, errG_G1, errG_G2,
                         resultD, resultD_G1, resultD_G2, 
                         errD_fake1+errD_fake2+errD_real, errG_G1+errG_G2, errD_L1])

        if np.mod(counter, 100) == 1:
          samples1, d_loss, g_loss = self.sess.run(
            [self.sampler1, self.d_loss, self.g_loss],
            feed_dict={
              self.z1: sample_z1,
              self.z2: sample_z2,
              self.inputs: sample_inputs,
            },
          )
          save_images(samples1, [8,8],
                './{}/train_G1_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          
          samples2, d_loss, g_loss = self.sess.run(
            [self.sampler2, self.d_loss, self.g_loss],
            feed_dict={
              self.z1: sample_z1,
              self.z2: sample_z2,
              self.inputs: sample_inputs,
            },
          )
          save_images(samples2, [8,8],
                './{}/train_G2_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            
          save_images((samples1+samples2)/2, [8,8],
                './{}/train_C_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

        if np.mod(counter, 500) == 2 and config.checkpoint_dir != 'checkpoint_dir':
          self.save(config.checkpoint_dir, counter)
          np.savetxt('./{}/loss_values_{}'.format(config.sample_dir, time.strftime("%m%d%H%M", time.localtime())) 
           + '.csv', self.loss_values, fmt= '%.4f', delimiter = ',', header=self.fileTitle, comments='')

    np.savetxt('./{}/loss_values_{}'.format(config.sample_dir, time.strftime("%m%d%H%M", time.localtime())) 
           + '.csv', self.loss_values, fmt= '%.4f', delimiter = ',', header=self.fileTitle, comments='')
#method-end

  def discriminator(self, image, y=None, reuse=False, name = "discriminator"):
    with tf.variable_scope(name) as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

      return tf.nn.sigmoid(h4), h4

  def generator(self, z, y=None, name = "generator", isTrain = True):
    with tf.variable_scope(name) as scope:
      if not isTrain: # reuse, sample mode
        scope.reuse_variables()
        
      if '1' in name:
        g_bn0 = self.g1_bn0
        g_bn1 = self.g1_bn1
        g_bn2 = self.g1_bn2
        g_bn3 = self.g1_bn3
      elif '2' in name:
        g_bn0 = self.g2_bn0
        g_bn1 = self.g2_bn1
        g_bn2 = self.g2_bn2
        g_bn3 = self.g2_bn3
      
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin_' + name, with_w=True)

      self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(g_bn0(self.h0, train=isTrain))

      self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1_'+name, with_w=True)
      h1 = tf.nn.relu(g_bn1(self.h1, train=isTrain))

      h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2_'+name, with_w=True)
      h2 = tf.nn.relu(g_bn2(h2, train=isTrain))

      h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3_'+name, with_w=True)
      h3 = tf.nn.relu(g_bn3(h3, train=isTrain))

      h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name=name+'g_h4_'+name, with_w=True)

      return tf.nn.tanh(h4)
      
  def generateSamples(self, config, filename1=None, filename2=None, filename3=None):
    for i in range(config.epoch):
      if not filename1:
        t = time.strftime("%m%d%H%M", time.localtime())
        filename1 = './{}/test_G1_{}_{}.mat'.format(config.sample_dir, i, t)
        filename2 = './{}/test_G2_{}_{}.mat'.format(config.sample_dir, i, t)
        filename3 = './{}/test_C_{}_{}.mat'.format(config.sample_dir, i, t)
        
      sample_z1 = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
      sample_z2 = sample_z1
      
      samples1 = self.sess.run(self.sampler1, feed_dict={self.z1: sample_z1, self.z2: sample_z2})
      save_images(samples1, [8,8], filename1)
      
      samples2 = self.sess.run(self.sampler2, feed_dict={self.z1: sample_z1, self.z2: sample_z2})
      save_images(samples2, [8,8], filename2)
      
      save_images((samples1+samples2)/2, [8,8], filename3)
      
      print("[Sampled]")

  @property
  def model_dir(self):
    return "{}_{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width,
        self.L1_lambda)
      
  def save(self, checkpoint_dir, step):
    if not checkpoint_dir:
      return
    
    model_name = "CoGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    if not checkpoint_dir:
      return False, 0
          
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
