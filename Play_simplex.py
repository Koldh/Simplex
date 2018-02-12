from pylab import *
from DL_agent import *
from Simplex import *
from Simplex_solver import *
import numpy
import cPickle
import time 

## 
# learning_mode parameter determines the way action's variables are considered:
#       - learning_mode= 0, then the action is selected among the variable having reduce_cost <0
#	- learning_mode = 1, with reduced_cost <= 0
#	- learning_mode = 2, among all non_basic variable

def playSimplex(LP='random',games_file_train=None,game_file_test= None,architecture='MLP',learning_mode=0,nb_repeat_game=5000,PARAMS=0):
	# Step 1: Init simplex
	feasible= 0
	EPOCH = nb_repeat_game
	converge = 0
	games_params = []
	SIMPLEX_results = []
	test_game_param = []
	SIMPLEX_results_test= []
	if LP == 'random':
		for i in xrange(len(games_file_train)):
			f = open(games_file_train[i],'rb')
			x = cPickle.load(f)
			f.close()
			c = x[0]
			A_ub = x[1]
			b_ub = x[2]
	                A_eq = None
	                b_eq = None
		        feasible,n_ite_simplex,obj_simplex,sol = TEST(c,A_ub,b_ub)
			size_T =  shape(A_ub)[0]
	        	seq_dim_1 = shape(c)[0]
       		 	seq_dim_0 = shape(A_ub)[0]
       		 	nb_sequence = seq_dim_1 + shape(A_ub)[0]
			games_params.append([c,A_ub,b_ub,A_eq,b_eq])
                        SIMPLEX_results.append([n_ite_simplex,obj_simplex])
	elif LP == 'TSP':
		for i in xrange(len(games_file_train)):
			f= open(games_file_train[i],'rb')
			x = cPickle.load(f)
			f.close()
			c = x[0]
			A_eq = x[1]
			b_eq = x[2]
			A_ub = x[3]
			b_ub = x[4]
			nb_constraint = shape(A_eq)[0]+shape(A_ub)[0]
			seq_dim_1 = shape(c)[0]
			input_dim = seq_dim_1
			seq_dim_0 = nb_constraint
			nb_sequence = input_dim + shape(A_ub)[0]
	                feasible,n_ite_simplex,obj_simplex,sol = TEST(c,A_ub,b_ub,A_eq,b_eq)
                        games_params.append([c,A_ub,b_ub,A_eq,b_eq])
			SIMPLEX_results.append([n_ite_simplex,obj_simplex])
	if game_file_test != None:
			for i in xrange(len(game_file_test)):
                        	f= open(game_file_test[i],'rb')
                        	x = cPickle.load(f)
				f.close()
                        	c = x[0]
                        	A_eq = x[1]
                        	b_eq = x[2]
				if LP == 'random':
		                        c = x[0]
        		                A_ub = x[1]
        		                b_ub = x[2]
        		                A_eq = None
        		                b_eq = None
	                                nb_constraint = shape(A_ub)[0]
				else:
		                        c = x[0]
        		                A_eq = x[1]
                		        b_eq = x[2]
                        		A_ub = x[3]
                        		b_ub = x[4]
                        		nb_constraint = shape(A_eq)[0]+shape(A_ub)[0]
                        	seq_dim_1 = shape(c)[0]
                        	input_dim = seq_dim_1
                        	seq_dim_0 = nb_constraint
                        	nb_sequence = input_dim + shape(A_ub)[0]
                        	feasible,n_ite_simplex,obj_simplex,sol = TEST(c,A_ub,b_ub,A_eq,b_eq)
                        	test_game_param.append([c,A_ub,b_ub,A_eq,b_eq])
                        	SIMPLEX_results_test.append([n_ite_simplex,obj_simplex])
	result_games =[]
	agent = BrainDQN(input_dim,nb_sequence,seq_dim_0,seq_dim_1,architecture,*PARAMS)
	print PARAMS
        obj = zeros((len(games_file_train),EPOCH))
        n_ite = zeros((len(games_file_train),EPOCH))
	n_ite_test = zeros((len(game_file_test),EPOCH))
	obj_test = zeros((len(game_file_test),EPOCH))
	for i in xrange(EPOCH):
		for k in xrange(len(games_file_train)):
			n_ite[k,i],obj[k,i],reward = train_game(games_params[k][0],games_params[k][1],games_params[k][2],learning_mode,games_params[k][3],games_params[k][4],architecture,agent,nb_sequence)
                	print 'GAME NUMBER ',k,'/ EPOCH NUMBER ', i, ' / NUMBER ITE ',n_ite[k,i],' / TIME STEP ',agent.timeStep, ' / OBJECT FUNC DQL', obj[k,i], ' / OBJECT FUNC SIMPLEX', SIMPLEX_results[k][1], ' / REWARD', reward
                	print '\n'
		for j in xrange(len(game_file_test)):
                	n_ite_test[j,i],obj_test[j,i] = test_game(test_game_param[j][0],test_game_param[j][1],test_game_param[j][2],learning_mode,test_game_param[j][3],test_game_param[j][4],architecture,agent,nb_sequence)
	result_games     = [n_ite,obj,agent.error_batch,agent.grad_weight_mlp,sol]
	result_game_test = [n_ite_test,obj_test]
        return result_games,SIMPLEX_results,result_game_test,SIMPLEX_results_test

def train_game(c,A_ub,b_ub,learning_mode,A_eq,b_eq,architecture,agent,nb_sequence):
		simplex = SIMPLEX(c,A_ub,b_ub,learning_mode,A_eq,b_eq)
		# Step 2: Init Tableau: Simplex PHASE 1
		simplex.get_init_tableaux()
		solution = simplex.prepare_phase_2()
		n_ite = 0
		obj = 0
		converge = 0
		if architecture == "Bi-RNN":
			observation = simplex.T[:-1,:-1].T
		elif architecture == "MLP":
			observation =  simplex.T[:-1,:-1].reshape((-1))
		elif architecture == 'CNN':
	                observation =  (simplex.T[:-1,:-1]-mean(simplex.T[:-1,:-1]))/(simplex.T[:-1,:-1].max())
		agent.setInitState(observation)
		# Step 3: Play until convergence
		complete = False
		deter_action = -1
		while complete==False:
			if np.shape(simplex.non_basis) == ():
				deter_action = simplex.non_basis
				action = zeros((nb_sequence))
				action[deter_action] = 1
				reward,complete,status = simplex.play(n_ite,action,deter_action,solution)
       	                	n_ite +=1
				deter_action = -1
	                elif np.shape(simplex.non_basis) == (0,):
	                        reward =1./(n_ite+1)#-self.T[-1,-1]
	                        complete = True
       	                        status   = 0
				if n_ite  == 0:
					print 'ERROR: NO SIMPLEX PHASE 2 IS NEEDED'
			elif  np.shape(simplex.non_basis) != () and np.shape(simplex.non_basis) != (0,):
				action = agent.getAction(simplex.non_basis)
				reward,complete,status = simplex.play(n_ite,action,deter_action,solution)
				n_ite +=1
			if status==0:
				converge +=1
			if architecture == "Bi-RNN":
				agent.setPerception(simplex.T[:-1,:-1].T,action,reward,complete)
			elif architecture == "MLP":
	                        agent.setPerception(simplex.T[:-1,:-1].reshape((-1)),action,reward,complete)
			elif architecture == 'CNN':
	                        agent.setPerception((simplex.T[:-1,:-1]-mean(simplex.T[:-1,:-1]))/(simplex.T[:-1,:-1].max()),action,reward,complete)
	        obj  = -simplex.T[-1,-1]
		return n_ite,obj,reward

def test_game(c,A_ub,b_ub,learning_mode,A_eq,b_eq,architecture,agent,nb_sequence):
	n_ite = 0
	obj   = 0
	converge = 0
        simplex = SIMPLEX(c,A_ub,b_ub,learning_mode,A_eq,b_eq)
        # Step 2: Init Tableau: Simplex PHASE 1
        simplex.get_init_tableaux()
        solution = simplex.prepare_phase_2()
        if architecture == "Bi-RNN":
        	observation = simplex.T[:-1,:-1].T
        elif architecture == "MLP":
        	observation =  simplex.T[:-1,:-1].reshape((-1))
        elif architecture == 'CNN':
        	observation =  (simplex.T[:-1,:-1]-mean(simplex.T[:-1,:-1]))/(simplex.T[:-1,:-1].max())
        agent.setInitState(observation)
        # Step 3: Play until convergence
        complete = False
        deter_action = -1
        while complete==False:
        	if np.shape(simplex.non_basis) == ():
                	deter_action = simplex.non_basis
                        action = zeros((nb_sequence))
                        action[deter_action] = 1
                        reward,complete,status = simplex.play(n_ite,action,deter_action,solution)
                        n_ite +=1
                        deter_action = -1
                elif np.shape(simplex.non_basis) == (0,):
                	reward =1#-n_ite[i]#-self.T[-1,-1]
                	complete = True
                        status   = 0
                 	if n_ite == 0:
                        	print 'ERROR: NO SIMPLEX PHASE 2 IS NEEDED'
                elif  np.shape(simplex.non_basis) != () and np.shape(simplex.non_basis) != (0,):
                 	action = agent.getAction(simplex.non_basis)
                        reward,complete,status = simplex.play(n_ite,action,deter_action,solution)
                        n_ite +=1
                if status==0:
                	converge +=1
                if architecture == "Bi-RNN":
                	agent.setPerception(simplex.T[:-1,:-1].T,action,reward,complete,testing=1)
                elif architecture == "MLP":
                        agent.setPerception(simplex.T[:-1,:-1].reshape((-1)),action,reward,complete,testing=1)
                elif architecture == 'CNN':
                        agent.setPerception((simplex.T[:-1,:-1]-mean(simplex.T[:-1,:-1]))/(simplex.T[:-1,:-1].max()),action,reward,complete,testing=1)
        obj = -simplex.T[-1,-1]
	return n_ite,obj



