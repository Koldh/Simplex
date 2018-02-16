import cPickle
from pylab import *
import tensorflow as tf
import sys
import glob

def CNN(input_variable,batch_size):
	l1     = tf.layers.conv1d(input_variable,10,1,data_format='channels_first',activation=tf.sigmoid)
	l2     = tf.layers.conv1d(l1,10,1,data_format='channels_first',activation=tf.sigmoid)
	l3     = tf.reshape(l2,(batch_size,-1))
	l4     = tf.layers.dense(l3,64,activation=tf.tanh)
	return  tf.layers.dense(l4,4,activation=None)



def arangedata(data):
	x=[]
	y=[]
	for d in data:
		x.append(d[1])
		y += [int(a) for a in d[0]]
	return concatenate(x,axis=0),asarray(y).flatten()

files = sort(glob.glob('../../../Simplex/sandbox/DATA/DATA_TSP*.pkl'))
X_train,X_test,Y_train,Y_test = [],[],[],[]

for i in files:
	f    = open(i,'rb')
	data = cPickle.load(f)
	f.close()
	p	   =0.8
	p	   =permutation(len(data))[:int(len(data)*p)]
	data_train = [data[i] for i in p]
	data_test  = [data[i] for i in xrange(len(data)) if i not in p]
	
	x_train,y_train = arangedata(data_train)
	x_train /=  x_train.max()
	x_test,y_test = arangedata(data_test)
	x_test /= x_test.max()
	x_train[x_train == inf] = 0
	x_test[x_test == inf]   = 0

	X_train.append(x_train)
	Y_train.append(y_train)
	X_test.append(x_test)
	Y_test.append(y_test)

X_train = concatenate(X_train)
X_test  = concatenate(X_test)
Y_train = concatenate(Y_train)
Y_test  =  concatenate(Y_test)

figure()
subplot(131)
imshow(X_train[:,0,:],aspect='auto')
colorbar()
subplot(132)
imshow(X_train[:,1,:],aspect='auto')
colorbar()
subplot(133)
imshow(X_train[:,2,:],aspect='auto')
colorbar()

figure()
plot(Y_train)

input_shape = (50,3,X_train.shape[2])
config = tf.ConfigProto()
config.log_device_placement=True
session  = tf.Session(config=config)

with tf.device('/device:GPU:'+str(0)):
        x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
        y_            = tf.placeholder(tf.int32, shape=[input_shape[0]],name='y')
        prediction    = CNN(x,input_shape[0])
	loss          = tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=prediction)
	optimizer     = tf.train.AdamOptimizer(learning_rate=0.000005)
    	train_op      = optimizer.minimize(loss=loss)
	accu          = tf.reduce_mean(tf.cast(tf.equal(y_,tf.cast(tf.argmax(prediction,axis=1),tf.int32)),tf.float32))

stack_loss = []
stack_accu = []

session.run(tf.global_variables_initializer())

for i in xrange(3000):
	print i
	p = permutation(len(X_train))[:input_shape[0]]
	x_batch = X_train[p]
	y_batch = Y_train[p]
	session.run(train_op,feed_dict={x:x_batch.astype('float32'),y_:y_batch.astype('int32')})
	lo = session.run(loss,feed_dict={x:x_batch.astype('float32'),y_:y_batch.astype('int32')})
	ac = session.run(accu,feed_dict={x:x_batch.astype('float32'),y_:y_batch.astype('int32')})
	stack_loss.append(lo)
	stack_accu.append(ac)
	print stack_loss[-1]

figure()	
plot(stack_loss)



figure()
plot(stack_accu,'r')
show()
