import matplotlib as mp ;
mp.use("Qt4Agg") ;
import gzip, pickle,numpy as np, matplotlib.pyplot as plt ;
import numpy.random as npr
import sys ;
import tensorflow as tf  ;
from matplotlib.widgets import Button ;
import math ;
import os ;
# disabling TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnistPath = "./corr_CIFAR10.pkl.gz"


with gzip.open(mnistPath, 'rb') as f:
  #python3
  #tmp = pickle.load(f, encoding='bytes')
  # python2
  tmp = pickle.load(f) ;

  traind = tmp["data_train"].astype("float32") ;
  trainl = tmp["labels_train"].astype("float32") ;
  testd = tmp["data_test"].astype("float32") ;
  testl = tmp["labels_test"].astype("float32") ;
  prop = tmp["properties"] ;
  w = prop["dimensions"][0] ;
  h = prop["dimensions"][1] ;
  ch = prop["num_of_channels"] ;
  print ("traind_shape=",traind.shape, "W/H/C=", w,h,ch) ;

# TODO: examine data!
print("train data shape: ",traind.shape)
print("label data shape: ",trainl.shape)
numlabels = trainl.argmax(axis = 1)
plt.hist(numlabels)
plt.show()

#label shape
print("label shape: ",trainl.shape) #50000*10

# check for block sorting
# sind die labels nach classe sortiert? es wird fuer jede zeile das maximale wert genommen --> argmax auf achse 1
print("are them block sorted?", trainl.argmax(axis=1)) #ja da immer erste alle values 0, dann 1, dann so weiter

# re-shuffle if needed
# means randomize!; was machen die mini batches --> nehmen immer 100 samples (nur hier batchSize) um die Daten  zu filtern --> siehe batches
# muss getan werden, da sonst die batch size zu klein ist und nur eine klasse gelernt wuerde!
nrTrainSamples = traind.shape[0]
indices = np.arange(0,nrTrainSamples) # alle indices der trainsamples holen
numpy.random.shuffle(indices) # alles indices shufflen, muss nicht zugewiesen werden, da implace "ersetzend"
traind = traind[indices] # fancyIndexing: Warum nicht beide einzeln? Da sonst der index ein anderern waere bei data und labels
trainl = trainl[indices]
print ("shuffledIndices", trainl.argmax(axis=1))

# check class balance
#plt.hist(trainl.sum(axis=0)) # oder
plt.hist(trainl.argmax(axis=1))
plt.show() # zeigt, dass alle Klassen gleich verteilt sind! --> kein Problem

dmin = traind.min(axis = 0) # hole die kleinste Zeile! wegen traind axis 0
dmax = traind.max(axis = 0)
print("min von min: ", dmin.min()) # hole das min aus der kleinsten Zeile
print("max von max: ", dmax.max()) # max zischen 1 und 10000 --> hist plotten
plt.hist(dmin)
plt.show() # da die x achse auf 10000 geht, mindestens eine Zahl auf 10000 --> wieder runter 
plt.hist(dmax)
plt.show() #gleiche wie bei mins
print(dmax.argmax()) # finde index an welcher position 10000
print(dmin.argmax()) #selbe!
# adjust it to values of the others! also wntweder durch 10000 oder einfach auf 1 setzen
traind[:,1548] = 1 # auf INDEX 1548 auf 1 setzen for all 

plt.imshow(traind[10].reshape(32,32,3) # wieder auf farbbild bringen und Bild an Stell 10 plotten


# check component ranges
# correct component ranges if needed

# visualize image 10

sess = tf.Session();

## input layer
data_placeholder = tf.placeholder(tf.float32,[None,3072]) ; #32*32*3 --> Layer dimensions muessen komplett durchgerechnet werden
label_placeholder = tf.placeholder(tf.float32,[None,10]) ;

## reshape data tensor into NHWC format
dataReshaped=tf.reshape(data_placeholder, (-1,32,32,3)) ; #adjusted to input data shape

## Hidden Layer 1
# Convolution Layer with 32 fiters and a kernel size of 5
conv1 = tf.nn.relu(tf.layers.conv2d(dataReshaped,6, 5,name="H1")) ; # filter muessen immer angegeben sein, channel frei waehlbar --> stepsize wenn nicht angegebn =1, channels = 6 (32/5)
print (conv1) ;
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2 --> 28x28x6 --> maxpooling 2,2 --> 28/2, 28/2 --> nach max pooling: 14*14*6
a1 = tf.layers.max_pooling2d(conv1, 2, 2) ;
print (a1) ;

# width L1 : Weights vorher channel - kernel size +1 = 32 - 5 +1 = 28
# heigthL1 : some oben

## Hidden Layer 2
# Convolution Layer with 64 filters and a kernel size of 3; unten: 16 channel!, 5 ist filter size--> nur eines, da beide gleich--> niemand weis das allerdings
conv2 = tf.nn.relu(tf.layers.conv2d(a1, 16, 5,name="H2")) ; # selbe formel wie oben: 14-5+1 =10, height das selbe 16 channels, da frei waehlbar CHANNEL muessen immer anngegeben werden
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
a2 = tf.layers.max_pooling2d(conv2, 2, 2) ; # nach maxpooling: 5*5*16 --> da als naechstes in fully connected Layer muss diese geflatet werden!
print (a2) ;
a2flat = tf.reshape(a2, (-1,5*5*16)) ; # hier wird geflatet, das was ich errechnet habe!

## Hidden Layer 3
Z3 = 120 ;
# allocate variables
W3 = tf.Variable(npr.uniform(-0.01,0.01, [5*5*16,Z3]),dtype=tf.float32, name ="W3") ; # adjust weights for a2flat!
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
Z5 = 10 ; # muss nicht geaendert werden, da wir hier immer noch 10 labels haben
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

## train!!

# variables and constants related to the drawing of batches 
nrTrainSamples = traind.shape[0] ; # wenn der batch index zu gross wird gibt es probleme wenn die Anzahl der batch sizesueber anzahl an samples (50000, deshalb  block oben)
batchIndex = -1 ; # wird gemacht wenn Arbeitspeicher eingeschraenkt ist, da sonst alle bilder im arbeitspeicher liegen wuerden!
batchSize = 100 ; # hier immer 100 samples per batch
maxBatchIndex = nrTrainSamples // batchSize ;
nrEpochs = 0 ;


iteration = 0 ;
tMax = 2000;
for iteration in range(0,tMax):

  # if we have exceeded the size of traind, restart!
  if batchIndex >= maxBatchIndex: # hier wird also wieder oben angefangen, wenn der Index ueber der max Anzahl liegt
    batchIndex=-1 ;
    nrEpochs += 1;

  # draw batches
  batchIndex +=1 ; 
  dataBatch = traind[batchIndex * batchSize:(batchIndex+1) * batchSize] ; 
  labelBatch = trainl[batchIndex * batchSize:(batchIndex+1) * batchSize] ;

  # update CNN parameters
  sess.run(update, feed_dict = {data_placeholder: dataBatch, label_placeholder : labelBatch }) ;

  # compute loss and accuracy, only every 50 iterations to save time
  if iteration%50==0:
    acc, lossVal= sess.run([nrCorrect, loss], feed_dict =  {data_placeholder: dataBatch, label_placeholder : labelBatch }) ;
    testacc = sess.run(nrCorrect, feed_dict = {data_placeholder: testd, label_placeholder: testl})
    print ("epoch=", nrEpochs, "iteration=", iteration, ", acc=", float(acc), "loss=", lossVal, "testacc=",testacc) ;
