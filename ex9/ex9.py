import gzip, pickle;
from tensorflow.contrib.optimizer_v2.gradient_descent import GradientDescentOptimizer
with gzip.open('mnist.pkl.gz', 'rb') as f:
    ((traind,train1),(vald,vall), (testd,testl)) = pickle.load(f);
    traind = traind.astype("float32").reshape(-1,784);
    train1 = train1.astype("float32");
    testd = testd.astype("float32").reshape(-1,784);
    testl = testl.astype("float32");
import tensorflow as tf;
    
with tf.Session() as sess:
    
    # create variable w with 2 values
    # construct Loss L = w1hoch2 + w2hoch2
    # create GradientTexentOptimizer (learning-rate=1)
    # create gradient operation and extract gradient from result
    weights = tf.Variable([2.,2.],dtype=tf.float32, name="W");
    # loss = tf.add(tf.square(weights)); 
    loss = tf.reduce_sum(weights*weights);
    
    gradientOptimizer = GradientDescentOptimizer(learning_rate = 0.2);
    gradientResult = gradientOptimizer.compute_gradients(loss, var_list=[weights]);
    print(gradientResult[0][0]);
    updater = tf.assign_add(weights, -0.2 * gradientResult[0][0]);
    
    ## init all variables
    sess.run(tf.global_variables_initializer()) ;       
    
    i = 0
    for i in range(0, 10):
        grad = sess.run(gradientResult);
        newWeight = sess.run(weights);
        loss = sess.run(loss);
        sess.run(updater);
        print("grad", grad[0][0][0], grad[0][0][1]);
        print("weights", weights[0], weights[1]);
        print("loss", loss);
        #print(gradientResult);