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
    tfVar = tf.Variable(np.zeros([60000, 10]), dtype=tf.float32);
    tfTrain1 = tf.placeholder(tf.float32, shape=[None, 10]);
    
    # Initialize variables to values specified
    sess.run(tf.global_variables_initializer());
    tfRange = tf.range(0, 60000);
    tfRandom = tf.random_uniform([60000], 0, 10, dtype=tf.int32);
    tfStack = tf.stack([tfRange, tfRandom], axis=1);
    
    # Veraendern von Variablen an Stellen von tfStack werden mit tfOnes ersetzt
    tfScattered = tf.scatter_nd_update(tfVar, tfStack, tf.ones(shape=[60000])); 

    fdict = {tfTrain1: train1};
    result = sess.run(tfScattered, feed_dict = fdict);
    print(result[0:10]);
    
    #2
    # Den groessten Index der Labels und Predictions holen
    tfnumLabels = tf.argmax(tfTrain1, axis=1);
    tfnumPredictions = tf.argmax(tfScattered, axis=1);
    
    # Indizes vergleichen
    tfIsEqual = tf.equal(tfnumLabels, tfnumPredictions);
    
    # Fehler berechnen
    tfClassError = tf.reduce_mean(tf.cast(tfIsEqual, tf.float32));
    error = sess.run(tfClassError, feed_dict =  fdict);
    print("2: class error is: ", error);
    
    #3
    #meins
    #npIndices = np.arange(0, 60000);
    #x = tf.placeholder(tf.float32, [3]);
    #y = tf.placeholder(tf.float32, [3]);
    #firstVector = tf.distributions.Categorical(probs=x);
    #secondVector = tf.distributions.Categorical(probs=y);
    #kl =  tf.distributions.kl_divergence(firstVector, secondVector);
    #fdict2 = {x: {1, 0, 0}, y: {1, 0, 0}};
    #klResult = sess.run(kl, fdict2);
    label = tf.constant([1, 0, 0], dtype= tf.float32);
    prediction = tf.placeholder(shape=[3], dtype=tf.float32);
    
    crossE = -tf.reduce_sum(label * tf.log(prediction + 0.0000001));
    fdict2 = {prediction: [1, 0, 0]};
    resultPred = sess.run(crossE, fdict2);
    print("first pred: ", resultPred);
    
    fdict2 = {prediction: [0.95, 0.025, 0.025]};
    resultPred = sess.run(crossE, fdict2);
    print("second pred: ", resultPred);
    
    fdict2 = {prediction: [0.8, 0.1, 0.1]};
    resultPred = sess.run(crossE, fdict2);
    print("third pred: ", resultPred);
    
    fdict2 = {prediction: [0.6, 0.2, 0.2]};
    resultPred = sess.run(crossE, fdict2);
    print("fourth pred: ", resultPred);
    
    fdict2 = {prediction: [0.4, 0.5, 0.1]};
    resultPred = sess.run(crossE, fdict2);
    print("fifth pred: ", resultPred);
    
    fdict2 = {prediction: [0.1, 0.8, 0.1]};
    resultPred = sess.run(crossE, fdict2);
    print("sixth pred: ", resultPred);
    
    fdict2 = {prediction: [0.0, 1, 0.0]};
    resultPred = sess.run(crossE, fdict2);
    print("seventh pred: ", resultPred);
    