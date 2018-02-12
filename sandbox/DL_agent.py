import tensorflow as tf 
import numpy as np
from pylab import * 
import random
from collections import deque 
from tensorflow.contrib import rnn
# Hyper Parameters:
FRAME_PER_ACTION = 1
#GAMMA = 0.99#0.5 # decay rate of past observations
#OBSERVE = 10. # timesteps to observe before training
#EXPLORE = 20000#20000. # frames over which to anneal epsilon
#FINAL_EPSILON = .0#0.001 # final value of epsilon
#INITIAL_EPSILON = 1.0#0.01 # starting value of epsilon
#REPLAY_MEMORY = 10 #  previous transitions to remember & before to train
#BATCH_SIZE = 10 # size of minibatch
#UPDATE_TIME = 100#20
#INITIAL_LEARNING = 0.000000001 
#FINAL_LEARNING = 0.0000000001
#INITIAL_LEARNING = 0.01
#FINAL_LEARNING = 0.01
#ANNEAL_LEARNING = 200000
#EPOCH = 5
class BrainDQN:
	def __init__(self,actions,nb_sequence,seq_dim_0,seq_dim_1,architecture,GAMMA,OBSERVE,EXPLORE,FINAL_EPSILON,INITIAL_EPSILON,REPLAY_MEMORY,BATCH_SIZE,UPDATE_TIME,INITIAL_LEARNING,FINAL_LEARNING,ANNEAL_LEARNING,EPOCH):
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.BATCH_SIZE = BATCH_SIZE
		self.epsilon = INITIAL_EPSILON
		self.FINAL_EPSILON = FINAL_EPSILON
		self.learning_rate = INITIAL_LEARNING
		self.GAMMA = GAMMA
		self.OBSERVE =  OBSERVE
		self.EXPLORE = EXPLORE
		self.FINAL_EPSILON = FINAL_EPSILON
		self.INITIAL_EPSILON = INITIAL_EPSILON
		self.REPLAY_MEMORY = REPLAY_MEMORY
		self.UPDATE_TIME = UPDATE_TIME
		self.FINAL_LEARNING = FINAL_LEARNING
		self.ANNEAL_LEARNING = ANNEAL_LEARNING
		self.EPOCH  = EPOCH
		self.error_batch = []
		self.grad_weight_mlp = []
		#self.actions = actions
                self.seq_dim_0 = seq_dim_0
                self.seq_dim_1 = seq_dim_1
		self.nb_sequence = nb_sequence
		self.actions  = nb_sequence
		self.architecture = architecture
		# init Q network
		with tf.variable_scope('Normal'):
			self.stateInput,self.QValue,self.params = self.createQNetwork()
		with tf.variable_scope('Target'):
			self.stateInputT,self.QValueT,self.paramsT = self.createQNetwork()
		# saving and loading networks
		self.session = tf.InteractiveSession()
                self.copyTargetQNetworkOperation = [p.assign(q) for p,q in zip(self.paramsT,self.params)]
                self.createTrainingMethod()
                self.session.run(tf.global_variables_initializer())


	def createQNetwork(self):
		if self.architecture == 'Bi-RNN':
              		n_hidden = 4*self.seq_dim_0
			weights = tf.Variable(tf.random_normal([2*n_hidden,1]),name='MLP')
			biases =  tf.Variable(tf.random_normal([1]),name='MLP_b')
			stateInput = tf.placeholder("float",[None,self.nb_sequence,self.seq_dim_0])
			stateInput_unstack = tf.unstack(stateInput,self.nb_sequence,axis=1,name='input')
			lstm_fw_cell  = rnn.BasicLSTMCell(n_hidden)
			lstm_bw_cell  = rnn.BasicLSTMCell(n_hidden)
			outputs, _,_ = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,stateInput_unstack, dtype=tf.float32)
			QVal= tf.transpose(tf.squeeze(tf.tensordot(outputs,weights,axes=[[2],[0]])))+biases
			return stateInput,QVal,[tf.Variable(lstm_fw_cell.trainable_weights[0].initialized_value()),tf.Variable(lstm_fw_cell.trainable_weights[1].initialized_value()),tf.Variable(lstm_bw_cell.trainable_weights[0].initialized_value()),tf.Variable(lstm_bw_cell.trainable_weights[1].initialized_value()),tf.Variable(weights.initialized_value()),tf.Variable(biases.initialized_value())]
		elif self.architecture == 'MLP':
			x = tf.placeholder("float",[None,self.nb_sequence*self.seq_dim_0])
			n_hidden_1 = 2*self.seq_dim_0*self.nb_sequence
			layer_1 = tf.nn.relu(tf.layers.dense(x,n_hidden_1))
			n_hidden_2 = 2*n_hidden_1
			layer_2 = tf.nn.relu(tf.layers.dense(layer_1,n_hidden_2))
                        weights = tf.Variable(tf.random_normal([n_hidden_2,self.nb_sequence]),name='MLP')
                        biases =  tf.Variable(tf.random_normal([self.nb_sequence]),name='MLP_b')
			QVal=  tf.squeeze(tf.matmul(layer_2,weights) + biases)
			return x,QVal,tf.trainable_variables()
		elif self.architecture == 'CNN':	
			W_conv1 = self.weight_variable([self.seq_dim_0,1,1,128])
			b_conv1 = self.bias_variable([128])
	
			W_conv2 = self.weight_variable([1,1,128,64])
			b_conv2 = self.bias_variable([64])
	
			W_conv3 = self.weight_variable([1,1,64,64])
			b_conv3 = self.bias_variable([64])
	
			#W_fc1 = self.weight_variable([self.nb_sequence*self.seq_dim_0*(64),512])
			#b_fc1 = self.bias_variable([512])

			W_fc2 = self.weight_variable([64,1])
			b_fc2 = self.bias_variable([1])
	
			# input layer
	
			stateInput = tf.placeholder("float",[None,self.seq_dim_0,self.nb_sequence,1])
	
			# hidden layers
			h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1) + b_conv1)
		#	h_pool1 = self.max_pool_2x2(h_conv1)
			h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2) + b_conv2)
			h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3) + b_conv3)

			h_conv3_flat = tf.reshape(h_conv3,[-1,self.nb_sequence,64])
                        QValue= tf.squeeze(tf.tensordot(h_conv3_flat,W_fc2,axes=[[2],[0]]))#+b_fc2
			return stateInput,QValue,[W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc2,b_fc2]

	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)

	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None])
		self.Q_Action =  tf.reduce_sum(tf.multiply(self.QValue, self.actionInput),axis=1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - self.Q_Action))
		if self.architecture == 'MLP':
			## TSP PARAM: self.opt  = tf.train.MomentumOptimizer(0.0000005,0.001,use_nesterov=True)
			self.opt  = tf.train.MomentumOptimizer(self.learning_rate,0.001,use_nesterov=True)
		elif self.architecture == 'CNN':
                        self.opt  = tf.train.MomentumOptimizer(self.learning_rate,0.1)
	        self.gradient  = self.opt.compute_gradients(self.cost)
		self.trainStep = self.opt.minimize(self.cost)
		

	def trainQNetwork(self):
		
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory,self.BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]
		if self.architecture == 'Bi-RNN':
			nextState_batch = np.reshape(nextState_batch,(np.shape(nextState_batch)[0],self.nb_sequence,self.seq_dim_0))
			state_batch = np.reshape(state_batch,(np.shape(state_batch)[0],self.nb_sequence,self.seq_dim_0))
		elif self.architecture == 'MLP':
                        nextState_batch = np.reshape(nextState_batch,(np.shape(nextState_batch)[0],self.nb_sequence*self.seq_dim_0))
                        state_batch = np.reshape(state_batch,(np.shape(state_batch)[0],self.nb_sequence*self.seq_dim_0))

		elif self.architecture == 'CNN':
                        nextState_batch = np.reshape(nextState_batch,(np.shape(nextState_batch)[0],self.seq_dim_0,self.nb_sequence,1))
                        state_batch = np.reshape(state_batch,(np.shape(state_batch)[0],self.seq_dim_0,self.nb_sequence,1))
		# Step 2: calculate y 
		y_batch = []
		QValue_batch = np.zeros((self.BATCH_SIZE,self.actions))
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
		for i in range(0,self.BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + self.GAMMA * np.max(QValue_batch[i]))
		self.trainStep.run(feed_dict={self.yInput :y_batch,self.actionInput : action_batch,self.stateInput : state_batch})
#		print norm(self.session.run(tf.trainable_variables()[4]))#print norm(self.session.run(self.a))
		if self.architecture != 'CNN':
			error_batch = mean(self.session.run(self.cost,feed_dict={self.yInput : y_batch,self.actionInput : np.squeeze(np.array(action_batch)),self.stateInput : np.squeeze(np.array(state_batch))}))
			grad_weight =mean(self.session.run(self.gradient[0],feed_dict={self.yInput : y_batch,self.actionInput : np.squeeze(np.array(action_batch)),self.stateInput : np.squeeze(np.array(state_batch))}))
			self.grad_weight_mlp.append(linalg.norm(grad_weight))
		elif self.architecture == 'CNN':
                        error_batch = mean(self.session.run(self.cost,feed_dict={self.yInput : y_batch,self.actionInput : np.squeeze(np.array(action_batch)),self.stateInput : np.array(state_batch)}))
                        grad_weight =mean(self.session.run(self.gradient[0],feed_dict={self.yInput : y_batch,self.actionInput : np.squeeze(np.array(action_batch)),self.stateInput : np.array(state_batch)}))
                        self.grad_weight_mlp.append(linalg.norm(grad_weight))

                #print 'ACTION',self.session.run(self.actionInput,feed_dict={self.yInput : y_batch,self.actionInput : np.squeeze(np.array(action_batch)),self.stateInput : np.squeeze(np.array(state_batch))})
                #print 'QV',self.session.run(self.QValue,feed_dict={self.yInput : y_batch,self.actionInput : np.squeeze(np.array(action_batch)),self.stateInput : np.squeeze(np.array(state_batch))})
                #print 'QAC',self.session.run(self.Q_Action,feed_dict={self.yInput : y_batch,self.actionInput : np.squeeze(np.array(action_batch)),self.stateInput : np.squeeze(np.array(state_batch))})
                #print 'INPUTy',self.session.run(self.yInput,feed_dict={self.yInput : y_batch,self.actionInput : np.squeeze(np.array(action_batch)),self.stateInput : np.squeeze(np.array(state_batch))})
		return error_batch
		
	def setPerception(self,nextObservation,action,reward,terminal,testing=0):
		if testing == 1:
	                if self.architecture == 'Bi-RNN':
       	                	newState = np.reshape(nextObservation,(1,self.nb_sequence,self.seq_dim_0))
       	        	elif self.architecture == 'MLP':
       	                	newState = np.reshape(nextObservation,(1,self.nb_sequence*self.seq_dim_0))
                	elif self.architecture == 'CNN':
                        	newState = np.reshape(nextObservation,(1,self.seq_dim_0,self.nb_sequence,1))
		else:
			if self.architecture == 'Bi-RNN':
				newState = np.reshape(nextObservation,(1,self.nb_sequence,self.seq_dim_0))
			elif self.architecture == 'MLP':
				newState = np.reshape(nextObservation,(1,self.nb_sequence*self.seq_dim_0))
			elif self.architecture == 'CNN':
				newState = np.reshape(nextObservation,(1,self.seq_dim_0,self.nb_sequence,1))
			self.replayMemory.append((self.currentState,action,reward,newState,terminal))
			if len(self.replayMemory) > self.REPLAY_MEMORY:
				self.replayMemory.popleft()
			if self.timeStep > self.OBSERVE:
				error_epoch = []
	                        for i in xrange(self.EPOCH):
	                               error = self.trainQNetwork()
       	                               error_epoch.append(error)
       	                        state = "train"
       	                	self.error_batch.append(np.mean(error_epoch))
				#if self.timeStep % 1000 == 0:
       	                	print 'ERROR EPOCH',np.mean(error_epoch) , " / TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon
#		state = "observe"
#		error_epoch = []
#		if self.timeStep % REPLAY_MEMORY == 0 and self.timeStep !=0:
#			for i in xrange(EPOCH):
#				error = self.trainQNetwork()
#				error_epoch.append(error)
#				state = "train"
#			self.error_batch.append(np.mean(error_epoch))
#	                print 'ERROR EPOCH',np.mean(error_epoch) , " / TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon
			if self.timeStep % self.UPDATE_TIME == 0: 
				print 'COPY TARGET NETWORK'
				self.copyTargetQNetwork()
	                self.timeStep += 1
		self.currentState = newState

	def getAction(self,non_basis):
		QValue = self.QValue.eval(feed_dict= {self.stateInput:self.currentState})
		action = np.zeros(self.actions)
		#QValue =  np.reshape(QValue,[np.shape(QValue)[0]*np.shape(QValue)[1]])
		size_non_basis = np.shape(non_basis)[0]
		if random.random() <= self.epsilon:
			action_non_basis_index = np.random.randint(size_non_basis)
			action_index = non_basis[action_non_basis_index]
			action[action_index] = 1
		else:
			action_index = non_basis[np.argmax(QValue[non_basis])]
			action[action_index] = 1

		# change episilon
		if self.epsilon > self.FINAL_EPSILON and self.timeStep > self.OBSERVE:
			self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON)/self.EXPLORE
		if self.learning_rate > self.FINAL_LEARNING and self.timeStep > self.EXPLORE:
			self.learning_rate -= (self.INITIAL_LEARNING-self.FINAL_LEARNING)/self.ANNEAL_LEARNING

		return action

	def setInitState(self,observation):
		if self.architecture == "Bi-RNN":
			self.currentState = np.reshape(observation,(1,self.nb_sequence,self.seq_dim_0))
		elif self.architecture == "MLP":
                        self.currentState = np.reshape(observation,(1,self.nb_sequence*self.seq_dim_0))
		elif self.architecture == 'CNN':
                        self.currentState = np.reshape(observation,(1,self.seq_dim_0,self.nb_sequence,1))
	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)
	def conv2d(self,x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
