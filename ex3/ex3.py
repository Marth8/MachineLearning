import gzip, pickle;
with gzip.open('mnist.pkl.gz', 'rb') as f:
    ((traind,train1),(vald,vall), (testd,testl)) = pickle.load(f);
    traind = traind.astype("float32").reshape(-1, 28, 28);
    train1 = train1.astype("float32");
    testd = testd.astype("float32").reshape(-1, 28, 28);
    testl = testl.astype("float32");
import numpy as np;
import matplotlib.pyplot as plt;
import numpy.random as npr;
import tensorflow as tf;

dataSlice1000 = tf.placeholder(tf.float32, shape=[1, 28, 28]) ;
dataVector = tf.placeholder(tf.float32, shape=[1, 10]);

with tf.Session() as sess:
    #fdict = {dataSlice1000: traind.slice([999, 0, 0], [1, 28, 28]), dataVector: train1.slice([999, 0, 0], [1, 10])};
    #npRes = sess.run(feed_dict = fdict);
    #print npRes;
    print "yes"