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

dataD = tf.placeholder(tf.float32, shape=[None, 28, 28]) ;
data1 = tf.placeholder(tf.float32, shape=[None, 10])
tfSample1000 = dataD[999];
tfClass1000 = data1[999];
with tf.Session() as sess:
    # Aufgabe a
    print ("\nAufgabe a \n");
    fdict = {dataD: traind, data1: train1};
    npRes = sess.run(tfSample1000, feed_dict = fdict);
    print (npRes.shape);
    print "yes"
    npRes = sess.run(tfClass1000, feed_dict = fdict);
    print (npRes.shape);
    [sample1000, label1000] = sess.run([tfSample1000, tfClass1000], feed_dict = fdict);
    
    # Aufgabe b
    print ("\nAufgabe b");
    tfNumLabels = tf.argmax(data1, axis = 1);
    npRes = sess.run(tfNumLabels, feed_dict = fdict);
    print (npRes);
    lowestLabel = tf.reduce_min(tfNumLabels);
    npRes = sess.run(lowestLabel, feed_dict = fdict);
    print (npRes);
    highesLabel = tf.reduce_max(tfNumLabels);
    npRes = sess.run(highesLabel, feed_dict = fdict);
    print (npRes);
    
    #Aufgabe c
    print ("\nAufgabe c");
    samplesPerClass = tf.reduce_sum(data1, axis = 0);
    samplesClass9 = samplesPerClass[9];
    npRes = sess.run(samplesClass9, feed_dict = fdict);
    print (npRes);
    
    #Aufgabe d
    print("\nAufgabe d");
    data10 = dataD[9];
    minData10 = tf.reduce_min(data10);
    maxData10 = tf.reduce_max(data10);
    min10, max10 = sess.run([minData10, maxData10], fdict);
    print("1d: ", min10, max10)
    
    # Aufgabe e
    print("\nAufgabe e (die vorletzten zwei machen kein Sinn, da Tensor immer immediate results sind)");
    tfMod1 = data10[::2];
    tfMod2 = data10[:,::2];
    tfInverse = data10[::-1,::-1];
    tfInverse2 = data10[::-2,::-2];
    [mod1, mod2, inverse, inverse2] = sess.run([tfMod1, tfMod2, tfInverse, tfInverse2], fdict);
    print("1e: ", mod1.shape, mod2.shape, inverse.shape, inverse2.shape);
    
    # Aufgabe f
    print("\nAufgabe f");
    tfNumLabels = tf.argmax(data1, axis = 1);
    tfMaskClass4 = tf.equal(tfNumLabels, 4);
    tfSamplesClass4 = tf.boolean_mask(dataD, tfMaskClass4);
    class4 = sess.run(tfSamplesClass4, fdict);
    print("1f: ", class4.shape);
    
    # Aufgabe g
    print("\nAufgabe g");
    tfNumLabels = tf.argmax(data1, axis = 1);
    tfMaskClass4 = tf.equal(tfNumLabels, 4);
    tfMaskClass9 = tf.equal(tfNumLabels, 9);
    tfMaskClass49 = tf.logical_or(tfMaskClass4, tfMaskClass9);
    tfSamplesClass49 = tf.boolean_mask(dataD, tfMaskClass49);
    class49 = sess.run(tfSamplesClass49, fdict);
    print("1g: ", class49.shape);

    # Aufgabe h (same wie numpy)
    print("\nAufgabe h");
    tfFirst10000 = dataD[:10000:];
    first10000 = sess.run(tfFirst10000, fdict);
    print("1h: ", first10000.shape);
    
    # Aufgabe i (same wie numpy)
    print("\nAufgabe i");
    indices = tf.range(60000);
    indices = tf.random_shuffle(indices);
    indices = indices[0:1000];
    randomData = tf.gather(dataD, indices);
    random1000Data = sess.run(randomData, fdict);
    print("1i: ", random1000Data.shape);
    