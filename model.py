import tensorflow as tf
import logging

def main(unused_argv):
    tf.logging.info("hello world")

if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    tf.app.run()
