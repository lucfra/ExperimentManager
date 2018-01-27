from tensorflow.python.platform import flags
import experiment_manager as em
import tensorflow as tf


flags.DEFINE_boolean('tst1', True, 'some doc string for tst1')
flags.DEFINE_float('bcnab', 1.2, 'other string')


def main(args=None):
    # print(flags.FLAGS.tst1, flags.FLAGS.bcnab)
    for key, value in tf.flags.FLAGS.__flags.items():
        print(key, value)
    pass

if __name__ == '__main__':
    tf.app.run()
