import cPickle
from pylab import *
import tensorflow as tf
import sys
import glob

def CNN(input_variable,batch_size):
	l1     = tf.layers.conv1d(input_variable,2**5,1,data_format='channels_first',activation=tf.nn.relu)
	l2     = tf.layers.conv1d(l1,2**4,1,data_format='channels_first',activation=tf.nn.relu)
	l3     = tf.reshape(l2,(batch_size,-1))
        l4     = tf.layers.dense(l3,2**6,activation=tf.nn.relu)
        l5     = tf.layers.dense(l4,2**5,activation=tf.nn.relu)
	return  tf.layers.dense(l4,4,activation=None)

def MLP(input_variable,batch_size):
	l1     = tf.reshape(input_variable,(batch_size,-1))
	l2     = tf.layers.dense(l1,2**7,activation=tf.nn.relu)
	l3     = tf.layers.dense(l2,2**6,activation=tf.nn.relu)
	l4     = tf.layers.dense(l3,2**5,activation=tf.nn.relu)
        l5     = tf.layers.dense(l4,2**4,activation=tf.nn.relu)
	return tf.layers.dense(l5,4,activation=None)


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
imshow((X_train[Y_train==0,0,:]),aspect='auto')
xlabel('Variables',fontsize=15)
ylabel(r'LP $n^{\circ}$',fontsize=15)
title('Dantzig Feature',fontsize=20)

subplot(132)
imshow((X_train[Y_train==1,1,:]),aspect='auto')
xlabel('Variables',fontsize=15)
ylabel(r'LP $n^{\circ}$',fontsize=15)
title('Steepest Edge Feature',fontsize=20)

subplot(133)
imshow((X_train[Y_train==2,2,:]),aspect='auto')
xlabel('Variables',fontsize=15)
ylabel(r'LP $n^{\circ}$',fontsize=15)
title('Greatest Improvement Feature',fontsize=20)


input_shape = (None,3,X_train.shape[2])
config = tf.ConfigProto()
config.log_device_placement=True
session  = tf.Session(config=config)

with tf.device('/device:GPU:'+str(0)):
        x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
        y_            = tf.placeholder(tf.int32, shape=[input_shape[0]],name='y')
        prediction    = CNN(x,input_shape[0])
	loss          = tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=prediction)
	optimizer     = tf.train.AdamOptimizer(learning_rate=0.005)#.0015 MLP
    	train_op      = optimizer.minimize(loss=loss)
	accu          = tf.reduce_mean(tf.cast(tf.equal(y_,tf.cast(tf.argmax(prediction,axis=1),tf.int32)),tf.float32))


stack_loss = []
stack_accu = []

session.run(tf.global_variables_initializer())

for i in xrange(80):
	p = permutation(len(X_train))[:input_shape[0]]
	x_batch = X_train[p]
	y_batch = Y_train[p]
	session.run(train_op,feed_dict={x:x_batch.astype('float32'),y_:y_batch.astype('int32')})
	lo = session.run(loss,feed_dict={x:x_batch.astype('float32'),y_:y_batch.astype('int32')})
	ac = session.run(accu,feed_dict={x:x_batch.astype('float32'),y_:y_batch.astype('int32')})
	stack_loss.append(lo)
	stack_accu.append(ac)
	if i%100 == 0:
		print 'iteration n_: ', i, '   Loss=  ', stack_loss[-1]



cpt=DNNplay(PATH,['Dantzig','Bland','Steepest','Greatest'],session,prediction,x)



figure()	
plot(stack_loss)



figure()
plot(stack_accu,'r')
show()
