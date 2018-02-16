import cPickle
from pylab import *
import tensorflow as tf



class CNN(self,input_variable):
                l1     = tf.layers.conv1d(input_variable,10,1,data_format='channels_first')
                l2     = tf.layers.conv1d(l1,10,1,data_format='channels_first')
                l3     = tf.layers.flatten(l2)
                return  tf.layers.dense(l3,4,activation=None)



def arangedata(data):
	x=[]
	y=[]
	for d in data:
		x.append(d[1])
		y.append([int(a) for a in d[0]])
	return concatenate(x,axis=0),asarray(y)


f = open('DATA_TSP.pkl','rb')
data = cPickle.load(f)
f.close()

p=0.5

p=permutation(len(data))[:int(len(data)*p)]
data_train = [data[i] for i in p]
data_test  = [data[i] for i in xrange(len(data)) if i not in p]

x_train,y_train = arangedata(data_train)
x_test,y_test = arangedata(data_test)


input_shape = (20,1,3)
config.log_device_placement=True
session  = tf.Session(config=config)

with tf.device('/device:GPU:'+str(gpu)):
        x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
        y_            = tf.placeholder(tf.int32, shape=[input_shape[0]],name='y')
        prediction    = CNN(x)
	loss          = tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=prediction)
	optimizer     = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    	train_op      = optimizer.minimize(loss=loss)

session.run(tf.global_variables_initializer())

for i in xrange(10):
	p = permutation(len(x_train))[:input_shape[0]]
	x_batch = x_train[p]
	y_train = y_train[p]
	print session.run(train_op,feed_dict={x:x_batch.astype('float32'),y_:y_batch.astype('int32')})







