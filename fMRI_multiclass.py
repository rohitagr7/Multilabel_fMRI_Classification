# A simple CNN to predict certain characteristics of the human subject from MRI images.
# This takes in a multilabel problem, convert to multiclass and then applies 3D convolution for classification
# Adjust needed for your dataset e.g., max pooling, convolution parameters, training_step, batch size, etc


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Parameters of fMRI images
width = 26
height = 31
depth = 23
#There are total of 45 unique labels in the data
nLabel = 45
#Conv layer params
num_channel_1 = 32
num_channel_2 = 64
#Training params
batch_size = 32
batch_index = 0
# Number of epochs
folds = 670
WEIGHT_DECAY = 0.01
_BATCH_NORM_DECAY=0.997
_BATCH_NORM_EPSILON=1e-5
_INITIAL_LEARNING_RATE = 1 * batch_size / (4*batch_size)

# Start TensorFlow InteractiveSession
#import input_3Dimage
import tensorflow as tf
import numpy as np
import random
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()

X_test = np.load('valid_test_X.npy')
X = np.load('train_X.npy')
Y = np.load('train_binary_Y.npy')

#If required to split the training data into train and validation set
#Training with full data was giving better result
"""
sf = ShuffleSplit(n_splits=1, test_size=0.2)
for train_indices,val_indices in sf.split(Y):
  X_train, Y_train = X[train_indices], Y[train_indices]
  X_val, Y_val = X[val_indices], Y[val_indices]
"""

X_train, Y_train = X, Y

N = len(Y_train)


#This function converts the k-hot labels to decimal for mapping
def get_decimal(x):
  dec =0
  j = 0
  n = len(x) - 1
  for i in range(len(x)):
    dec += int(x[n-i])*(2**j)
    j += 1
  return dec

def create_mapping(y):
  unique = []
  for i in range(len(y)):
    unique.append(get_decimal(y[i]))
  return unique

lb = preprocessing.LabelBinarizer()
unique = create_mapping(Y)
lb.fit(unique)

"""
y = create_mapping(Y_val)
y_val = lb.transform(y)

"""

# Placeholders 
x = tf.placeholder(tf.float32, shape=[None, width,height,depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 10]

## Weight Initialization
# Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
def weight_variable(shape):
  initial = tf.truncated_normal(shape,mean=0, stddev=0.1)
  return tf.Variable(initial, trainable=True)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, trainable=True)

## Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


## First Convolutional Layer
# Conv then Max-pooling. 1st layer will have 32 features for each 5x5 patch. (1 feature -> 32 features)
W_conv1 = weight_variable([3, 3, 3, 1, num_channel_1])  # shape of weight tensor = [5,5,1,32]
b_conv1 = bias_variable([num_channel_1])  # bias vector for each output channel. = [32]

# Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
x_image = tf.reshape(x, [-1,width,height,depth,1]) # [-1,28,28,1]

# x_image * weight tensor + bias -> apply ReLU -> apply max-pool
layer_1 = conv3d(x_image, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(layer_1)  # conv2d, ReLU(x_image * weight + bias)
h_pool1 = max_pool_2x2(h_conv1)  # apply max-pool 


## Second Convolutional Layer
# Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
W_conv2 = weight_variable([2, 2, 2, num_channel_1, num_channel_2]) # [5, 5, 32, 64]
b_conv2 = bias_variable([num_channel_2]) # [64]

layer_2 = conv3d(h_pool1, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(layer_2)  # conv2d, .ReLU(x_image * weight + bias)
h_pool2 = max_pool_2x2(h_conv2)  # apply max-pool 


## Densely Connected Layer (or fully-connected layer)
# fully-connected layer with 1024 neurons to process on the entire image
W_fc1 = weight_variable([7*8*6*num_channel_2, 1024])  # [7*7*64, 1024]
b_fc1 = bias_variable([1024]) # [1024]]
#W_fc2 = weight_variable([1024, 512])  # [7*7*64, 1024]
#b_fc2 = bias_variable([512]) # [1024]]

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*8*6*num_channel_2])  # -> output image: [-1, 7*7*64] = 3136
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#2nd dense layer (if required)
#h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # ReLU(h_pool2_flat x weight + bias)
#h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
## Readout Layer
W_final = weight_variable([1024, nLabel]) # [1024, 10]
b_final = bias_variable([nLabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, W_final) + b_final


global_step = tf.train.get_or_create_global_step()


## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
loss = cross_entropy + WEIGHT_DECAY* tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

optimizer = tf.train.AdamOptimizer(0.0000065)  # 1e-4
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_step = optimizer.minimize(loss, global_step)

probabilities = tf.nn.softmax(y_conv)
classes = tf.argmax(y_conv, axis=1)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), classes),tf.float32))

sess.run(tf.global_variables_initializer())

def get_data(sess,mode, batch_size):
    global batch_index
    if mode=='train':
      max = N
      begin = batch_index
      end = batch_index + batch_size
      if end >= max:
          end = max
          batch_index = 0

      x_data = X_train[begin:end,:,:,:]
      y = Y_train[begin:end,:]

      label = create_mapping(y)
      #Transform multilabel to multiclass
      Y = lb.transform(label)

      batch_index += batch_size  # update index for the next batch
      return x_data, Y
    elif mode=='val':
      return(X_val, y_val)
    else:
      return(X_test)
      




# Include keep_prob in feed_dict to control dropout rate.
for it in range(folds):
  print("\n\nStarting training iteration ", it)
  for i in range(N//batch_size):
      batch = get_data(sess, 'train',batch_size)
      if i%20 == 0:
          train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
          train_loss = loss.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
          y_conv1 = y_conv.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
          gs = global_step.eval()
          print("step %d global_step %d  training accuracy %g loss %f"%(i, gs, train_accuracy, train_loss))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


#Evaluating on validation set(if required)
"""
  print("\n\nValidation set# ", it)
  val_batch = get_data(sess, 'val', batch_size)

  val_acc = accuracy.eval(feed_dict={x: val_batch[0],y_: val_batch[1], keep_prob:1.0})
  print('Validation Accuracy::',val_acc)
"""

#Dumping the output of conv layers
"""
train_dump = []
for i in range(len(X)):
  p,q,r = np.shape(X[i])
  testdata = np.reshape(X[i],(-1,p,q,r))
  conv_output = h_pool2.eval(feed_dict={x:testdata, keep_prob: 1.0})[0]
  train_dump.append(conv_output)
  
np.save('conv_output_train_64.npy',train_dump)
test_dump = []
for i in range(len(X_test)):
  p,q,r = np.shape(X_test[i])
  testdata = np.reshape(X_test[i],(-1,p,q,r))
  conv_output = h_pool2.eval(feed_dict={x:testdata, keep_prob: 1.0})[0]
  test_dump.append(conv_output)
  
np.save('conv_output_test_64.npy',test_dump)
"""

#Generate labels for test data
print("\n\nRunning test set")
testset = get_data(sess,'test',batch_size)
result = []
for i in range(len(testset)):
  p,q,r = np.shape(testset[i])
  testdata = np.reshape(testset[i],(-1,p,q,r))
  pred = probabilities.eval(feed_dict={x: testdata, keep_prob:1.0})[0]
  result.append(np.argmax(pred))
result = np.array(result)
#Converting the decimal results back to k-hot representation
result_one_hot = np.zeros((result.size, nLabel))
result_one_hot[np.arange(result.size),result] = 1
dec = lb.inverse_transform(result_one_hot)
binary = [np.binary_repr(item, width=19) for item in dec]
test_result = np.array([list(item) for item in binary], dtype=int)

np.save("test_result_CNN_multiclass.npy", test_result)
