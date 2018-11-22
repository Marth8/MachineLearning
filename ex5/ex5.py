import gzip, pickle;
with gzip.open('mnist.pkl.gz', 'rb') as f:
    ((traind,train1),(vald,vall), (testd,testl)) = pickle.load(f);
    traind = traind.astype("float32").reshape(-1, 28, 28);
    train1 = train1.astype("float32");
    testd = testd.astype("float32").reshape(-1, 28, 28);
    testl = testl.astype("float32");
import numpy as np;
import tensorflow as tf;

with tf.Session() as sess:
    tfTrain1 = tf.placeholder(tf.float32, shape=[None, 10]);
    tfTraind = tf.placeholder(tf.float32, shape=[None, 28, 28]);
    
    # exercise 1
    tfData = tf.placeholder(tf.float32, shape=[None, None]);
    tfExp = tf.exp(tfData)
    tfSm =  tf.divide(tfExp, tf.reduce_sum(tfExp, axis=1, keep_dims=True));

    # Initialize variables to values specified;
    data = [[0.5, 0, 1], [0.5, 0.5, 0.5], [10, 10, 10], [0.5, -1, -2]];
    fdict = {tfData: data};
    result = sess.run(tfSm, feed_dict = fdict);
    print result;
    
    tfSmSum = tf.reduce_sum(tfSm, axis = 1);
    smSum = sess.run(tfSmSum, feed_dict = fdict);
    print smSum;
    
    # simple softmax
    print sess.run(tf.nn.softmax(tfData), feed_dict = fdict);
    
    # exercise 2
    
    # reshape traind
    tfReshaped = tf.reshape(tfTraind, shape=(60000, 784));
    print tfReshaped.shape;
    
    # tfReshaped muss zu Vektor (60000, 10), dadurch (60000, 784) * (10, 784) => Matrix on (60000, 10)
    tfB = tf.Variable(np.ones([1, 10]), dtype=tf.float32);
    tfW = tf.Variable(np.zeros([10, 784]), dtype=tf.float32);
    sess.run(tf.global_variables_initializer());
    
    # bias has 1,10 because of broadcast
    tfReshapedMat= tf.matmul(tfReshaped, tf.transpose(tfW)) + tfB;
    reshapedMat = sess.run(tfReshapedMat, feed_dict = {tfTraind: traind});
    tfMnistExp = tf.exp(tfReshapedMat);
    tfMnistSm = tf.divide(tfMnistExp, tf.reduce_sum(tfMnistExp, axis=1, keep_dims=True));
    mnistSm = sess.run(tfMnistSm, feed_dict = {tfTraind: traind});
    print mnistSm;
    print sess.run(tf.reduce_sum(tfMnistSm, axis = 1), feed_dict = {tfTraind: traind});
    
    # exercise 3
    confusion = [[50, 10, 20, 20], [10, 20, 60, 10], [10, 30, 60, 0], [30, 0, 0, 70]];
    tfConfusion = tf.placeholder(tf.float32, shape=[None, 4]);
    tfNumberOfSamples = tf.reduce_sum(tf.reduce_sum(tfConfusion, axis = 1));
    confusionDict = {tfConfusion: confusion};
    numberOfSamples = sess.run(tfNumberOfSamples, feed_dict=confusionDict);
    print numberOfSamples;
    
    