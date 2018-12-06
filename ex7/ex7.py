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
    dataPlaceholder = tf.placeholder(tf.float32, shape=[None, 784]);
    labelPlaceholder = tf.placeholder(tf.float32, shape=[None, 10]);
    
    # 28 mal 28 mal 1 mit (5, 5) max Pooing 2x2 (1 Input of 28x28)
    # Dann 32 mit maxPooling 2x2 (3, 3) 
    # Dann 64 auf einen 1000 Vector und dann auf einen 10er(10 Klassen)
    # 5 Komponenten, 4 Übergänge
    # H(l) = (H(l-1) - fy(l))/ y(l)  + 1, W(l) = (W(l-1) - fx(l)) / x(l)  + 1 
    # Dadurch im ersten Layer: H(l) = 24(28 - 5 /1 + 1)) W(l) = 24
    # Dadurch ist der Zweite Layer: 24x24 vor dem Max-Pooling
    # Dann max-Pooling: 12x12x32 auf dem zweiten Layer (32 Inputs of 12x12)
    # Dann Layer 2: 12 - 3 = 9 / 1 = 10 => 10x10
    # Mit Max-Pooling = 5x5 => Layer 3 ist dann 5x5x64 (64 Inputs of 5x5)

    N = 10000;
    fd = { dataPlaceholder: traind[0:N], labelPlaceholder: tfTrain1[0:N]};

    # 1a (Data wird Input von 1 mit 28x28 => NHWC Format
    reshapedData = tf.reshape(dataPlaceholder, (-1, 28, 28, 1));
    print (reshapedData)
    
    ##Hidden Layer 1 numberOfChannels = 32 filters: 5x5
    conv1 = tf.nn.relu(tf.layers.conv2d(reshapedData, 32, 5, name="H1"));
    print (conv1);
    #max pooling von 2x2
    a1 = tf.layers.max_pooling2d(conv1, 2, 2);
    print(a1);
    
    ##Hidden Layer 2, 64 channels, 3x3 Filters
    conv2 = tf.nn.relu(tf.layers.conv2d(a1, 64, 3, name = "H2"));
    # max pooling 2x2
    a2 = tf.layers.max_pooling2d(conv2, 2, 2);
    print(a2);
    #flatten the layer
    a2flat = tf.reshape(a2, (1, 5*5*64));
    
    ##Hidden Layer 3
    Z3 = 300
    
    # allocate Variables
    #W3 = tf.Variable(np.uniform(-0.01, 0.01, [5*5*64, Z3]), dtype=tf.float32, name="W4");
    #B3 =  tf.Variable(np.uniform(-0.01, 0.01, [1, Z3]), dtype=tf.float32, name="B4");
    
    #compute activiation
    #logits = tf.matmul(a3, w4) +b4;
    #print(logits)
    