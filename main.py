# from train import model
from model import PIAFusion
import numpy as np
import tensorflow as tf

import pprint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 64, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 2, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-3, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("stride", 24, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("summary_dir", "log", "Name of log directory [log]")
flags.DEFINE_boolean("is_train",False, "True for training, False for testing [True]")
flags.DEFINE_string("model_type", 'PIAFusion', "Illum for training the Illumination Aware network,"
                                                 " PIAFusion for training the Fusion Network [classifier]")
flags.DEFINE_string("DataSet", 'TNO', "The Dataset for Testing, TNO, RoadScene, MSRS,  [TNO]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()
def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        piafusion = PIAFusion(sess,
                      image_size=FLAGS.image_size,
                      label_size=FLAGS.label_size,
                      batch_size=FLAGS.batch_size,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      model_type=FLAGS.model_type,
                      phase=FLAGS.is_train,
                      Data_set=FLAGS.DataSet)
        if FLAGS.is_train:
            piafusion.train(FLAGS)
        else:
            piafusion.test(FLAGS)


if __name__ == '__main__':
    tf.app.run()
