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

# 60000 -> 28 -> 28
print(traind.shape);

# 60000 -> 10 => 600000
print(train1.shape);

# 28
print(len(traind[0][0]));

# Es gibt 10 verschiedene Klassen.
print(train1[0][0]);

# Hoechste Klasse ist Klasse 6
print(train1[999].argmax());

test = traind[999];
plt.imshow(test);
plt.show();

#1.c
classVector = train1.argmax(axis=1);
print(classVector.min());
print(classVector.max());

#1.d Dimension wird durch sum entnommen
samplesPerClass = train1.sum(axis=0);
print("samplesPerClass", samplesPerClass);
print("samples of class", samplesPerClass[5]);

#1.e
sample10 = traind[9];
print("minPix", sample10.min());
print("maxPix", sample10.max());

#1.f
print("row/2", sample10[::2].shape)
print("column/2", sample10[::][:2].shape);
print("inverse", sample10[sample10.shape[0]:0:-1][sample10.shape[1]:0:-1].shape);
sample10[0:11] = 0;
sample10[sample10.shape[0]:sample10.shape[0] - 10:-1] = 0;
#1.g
classVector = train1.argmax(axis=1);
class4Mask = (classVector == 4);
traind_class4 = traind[class4Mask];
print(traind_class4.shape);

#1.h
class4or9Mask = np.logical_or((classVector == 9), class4Mask);
print(traind[class4or9Mask].shape)

#1.i
#reshapedTrain = np.reshape(traind[0:10000], train1[0:10000]);
#print reshapedTrain.shape
#concate = np.column_stack((traind[0:10000], train1[0:10000]));
#print(concate);

#1.j Use same variable cause of copy!!!
traind -= 1;

#1.k
indices = np.arange(0, traind.shape[0]);
#unordering the indizes
npr.shuffle(indices);
indices = indices[0:1000];
randomizedStack = traind[indices];
print(randomizedStack.shape)
plt.imshow(traind[500]);
plt.show();
