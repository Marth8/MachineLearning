# LeNet-5
import matplotlib as mp ;
mp.use("Qt4Agg") ;
import gzip, pickle,numpy as np, matplotlib.pyplot as plt ;
import numpy.random as npr, tensorflow as tf, sys  ;
from matplotlib.widgets import Button ;
import math ;

mnistPath = "./mnist.pkl.gz"


def test(x,y):
    #return ;
    x [:,[0,10,50,200,300,700]] *= 1000 ;
    num_y = y.argmax(axis=1) ;
    s_ind = np.argsort(num_y,axis=0) ;
    y = y[s_ind] ;
    x = x[s_ind] ;

sess = tf.Session();

dataImportHandler  = test ;
with gzip.open(mnistPath, 'rb') as f:
    # python3
    #tmp=pickle.load(f, encoding='bytes')
    # python2
    ((data,labels), (a,b), (c,d)) = pickle.load(f) ;
    # ignore!
    dataImportHandler(data,labels);
    traind = data ;
    trainl = labels ;
    testd = data ;
    testl = labels ;




# process and analyze numpy data
print (trainl.shape)
samplesPerClass = trainl.sum(axis = 0);
print (samplesPerClass);
argmax1 = trainl.argmax(axis = 1);
print (trainl[:4000].sum(axis=0));
print (argmax1);


samples = traind.argmax(axis = 1);
print (samples.min(axis = 0));
print (samples.max(axis = 0));
print (traind.min(axis = 0));
#print (traind.max(axis = 0));
print (traind.max(axis = 0).mean())
print (traind.max(axis = 0).var());
print (traind.max(axis=0).min());
print (traind.max(axis=0).max());
maxTraind = traind.max(axis=1);
print (traind[maxTraind > 900].shape); # groesser als 800
maxax0 = np.max(data, axis = 0, keepdims = True);
mask = (maxax0 > 4) * 0.001 + (maxax0 < 4) ; #[0.001, 1, 1, 1, ...., 0.0001]
traind *= mask;
plt.hist(data[200]);
plt.show();

# mittelwert von 4 und hoechster wert ist 1000 => grosse Varianz
# minimaler Wert vom Minimalen und vom Maximalen und den Maximalen Wert vom Minimalen und Maximalen
# mittelwert der maximal (normalisieren)
# datensatz ist sortiert (shufflen)

# wieder ordentlich machen: shufflen und normalisieren ( xi = (xi-minxi) / (max xi - minxi))
traind = (traind - traind.min(axis = 0)) / (traind.max(axis = 0) - traind.min(axis = 0)) 

# remove this when data are preprocessed
#sys.exit(0) ;



## input layer
data_placeholder = tf.placeholder(tf.float32,[None,784]) ;
label_placeholder = tf.placeholder(tf.float32,[None,10]) ;

## reshape data tensor into NHWC format
dataReshaped=tf.reshape(data_placeholder, (-1,28,28,1)) ;

## Hidden Layer 1
# Convolution Layer with 32 fiters and a kernel size of 5
conv1 = tf.nn.relu(tf.layers.conv2d(dataReshaped,6, 5,name="H1")) ;
print (conv1) ;
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
a1 = tf.layers.max_pooling2d(conv1, 2, 2) ;
print (a1) ;

## Hidden Layer 2
# Convolution Layer with 64 filters and a kernel size of 3
conv2 = tf.nn.relu(tf.layers.conv2d(a1, 16, 5,name="H2")) ;
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
a2 = tf.layers.max_pooling2d(conv2, 2, 2) ;
print (a2) ;
a2flat = tf.reshape(a2, (-1,4*4*16)) ;

## Hidden Layer 3
Z3 = 120 ;
# allocate variables
W3 = tf.Variable(npr.uniform(-0.01,0.01, [4*4*16,Z3]),dtype=tf.float32, name ="W3") ;
b3 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z3]),dtype=tf.float32, name ="b3") ;
# compute activations
a3 = tf.nn.relu(tf.matmul(a2flat, W3) + b3) ;
print (a3) ;

## Hidden Layer 4
Z4 = 84 ;
# allocate variables
W4 = tf.Variable(npr.uniform(-0.01,0.01, [Z3,Z4]),dtype=tf.float32, name ="W4") ;
b4 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z4]),dtype=tf.float32, name ="b4") ;
# compute activations
a4 = tf.nn.relu(tf.matmul(a3, W4) + b4) ;
print (a4) ;


## output layer
# alloc variables
Z5 = 10 ;
W5 = tf.Variable(npr.uniform(-0.1,0.1, [Z4,Z5]),dtype=tf.float32, name ="W5") ;
b5 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z5]),dtype=tf.float32, name ="b5") ;
# compute activations
logits = tf.matmul(a4, W5) + b5 ;
print (logits) ;

## loss
lossBySample = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_placeholder) ;
loss = tf.reduce_mean(lossBySample) ;

## classification accuracy
nrCorrect = tf.reduce_mean(tf.cast(tf.equal (tf.argmax(logits,axis=1), tf.argmax(label_placeholder,axis=1)), tf.float32)) ;

## create update op
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1) ;  # 0.00001
update = optimizer.minimize(loss) ;

## init all variables
sess.run(tf.global_variables_initializer()) ;

## learn!!
iteration = 0 ;
tMax = 1000;
batchIndex = 0 ;
batchSize = 100 ;

for iteration in range(0,tMax):
    # select a random set of 100 samples and labels
    indices = npr.uniform (0, traind.shape[0],[batchSize]).astype(np.int32) ;
    dataBatch = traind[ indices ] ;
    labelBatch = trainl[ indices ] ;

    # update parameters
    sess.run(update, feed_dict = {data_placeholder: dataBatch, label_placeholder : labelBatch }) ;

    # compute loss and accuracy
    acc, lossVal= sess.run([nrCorrect, loss], feed_dict =  {data_placeholder: dataBatch, label_placeholder : labelBatch }) ;
    testacc = sess.run(nrCorrect, feed_dict = {data_placeholder: testd, label_placeholder: testl})
    print ("epoch ", iteration, "acc=", float(acc), "loss=", lossVal, "testacc=",testacc) ;
















