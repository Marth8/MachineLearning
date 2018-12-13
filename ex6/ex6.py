import tensorflow as tf;
    
sess = tf.Session();
tf_data = tf.Variable(tf.random_uniform([10000,2],-0.5, 0.5, dtype=tf.float32), dtype=tf.float32);
tf_target = tf.Variable(tf.add(tf.reduce_sum(tf.pow(tf_data, 2), axis=1),tf.random_uniform([10000], -0.1, 0.1, dtype=tf.float32)), dtype=tf.float32);

b1 = tf.Variable(tf.random_uniform([1,20], -0.01, 0.01, dtype=tf.float32),dtype=tf.float32);
w1 = tf.Variable(tf.random_uniform([2,20], -0.01, 0.01, dtype=tf.float32),dtype=tf.float32);

b2 = tf.Variable(tf.random_uniform([1,20], -0.01, 0.01, dtype=tf.float32),dtype=tf.float32);
w2 = tf.Variable(tf.random_uniform([20,20], -0.01, 0.01, dtype=tf.float32),dtype=tf.float32);

b3 = tf.Variable(tf.random_uniform([1,1], -0.01, 0.01, dtype=tf.float32),dtype=tf.float32);
w3 = tf.Variable(tf.random_uniform([20,1], -0.01, 0.01, dtype=tf.float32),dtype=tf.float32);

sess.run(tf.global_variables_initializer());

hl1 = tf.nn.relu(tf.matmul(tf_data, w1) + b1);
hl2 = tf.nn.relu(tf.matmul(hl1, w2) + b2);
out = tf.matmul(hl2, w3) + b3;

#reduce((modeloutput-target)^2)

loss = tf.reduce_mean(tf.pow(tf.subtract(out, tf_target),2));

lout = sess.run(out);
print(lout.shape);

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1);
update = optimizer.minimize(loss) ;

iteration = 0 ;

for iteration in range(0,5000):

    sess.run(update) ;
    lossVal = sess.run(loss) ;
    print ("epoch ", iteration, "loss=", lossVal);
    
    
    