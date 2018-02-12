from pylab import *
from Play_simplex import *
import cPickle
import string
import sys

######################################### PARAMS ####################################################
games_file_train = [ 'TSPdata'+str(i)+'.txt' for i in range(1,21)]
game_file_test  = [ 'TSPdata'+str(i)+'.txt' for i in range(21,50)]

LP = 'TSP'
architecture = 'CNN'
learning_mode = 0 #learning_mode parameter determines the way action's variables are considered:
		  #       - learning_mode= 0, then the action is selected among the variable having reduce_cost <0
     		  #       - learning_mode = 1, with reduced_cost <= 0
		  #       - learning_mode = 2, among all non_basic variable

nb_repeat_game = 10000
disp_plot = 0

#GAMMA = 0.99#0.5 # decay rate of past observations
OBSERVE = 10. # timesteps to observe before training
EXPLORE = 50000#20000. # frames over which to anneal epsilon
FINAL_EPSILON = .0#0.001 # final value of epsilon
INITIAL_EPSILON = 1.0#0.01 # starting value of epsilon
REPLAY_MEMORY = 10 #  previous transitions to remember & before to train
BATCH_SIZE = 10 # size of minibatch
UPDATE_TIME = 100#20
#INITIAL_LEARNING = 0.000000001 
#FINAL_LEARNING = 0.0000000001
INITIAL_LEARNING = 0.01
FINAL_LEARNING = 0.001
ANNEAL_LEARNING = 200000
EPOCH = 5

for GAMMA in linspace(0.4,0.75,4):
	for REPLAY_MEMORY in [10,30,100]:
		for BATCH_SIZE in range(10,100,40):
			for UPDATE_TIME in range(50,500,200):
				for INITIAL_LEARNING in linspace(0.000001,0.001,3):
					for EPOCH in [5]:
						PARAMS = [GAMMA,OBSERVE,EXPLORE,FINAL_EPSILON,INITIAL_EPSILON,REPLAY_MEMORY,BATCH_SIZE,UPDATE_TIME,INITIAL_LEARNING,FINAL_LEARNING,ANNEAL_LEARNING,EPOCH]
						print PARAMS
#################################### RUN SIMPLEX GAME ##############################################
						result_games,result_simplex_train,result_game_test,result_simplex_test = playSimplex(LP,games_file_train,game_file_test,architecture,learning_mode,nb_repeat_game,PARAMS)


#print 'TEST RESULT: DL AGENT: nb_iteration= ',result_game_test[0] ,'obj= ', result_game_test[1],' / SIMPLEX HEURISTIC: nb_iteration= ',result_simplex_test[0][0],'obj= ',result_simplex_test[0][1]

						n_ite_games_train,obj,error_batch,grad_weight,sol  = result_games
						params_bis =  [str(i) for i in PARAMS]
						PARAMS_TITLE =  '_'.join(params_bis)
						n_ite_train	    = []
						for i in xrange(len(games_file_train)):
							n_ite_train.append([n_ite_games_train[i],result_simplex_train[i][0]])
						n_ite_test	    = []
						for i in xrange(len(game_file_test)):
							n_ite_test.append([result_game_test[0][i],result_simplex_test[i][0]])

						f = open('n_ite_'+PARAMS_TITLE+'.pkl','wb')
						cPickle.dump([n_ite_train,n_ite_test],f)
						f.close()
						tf.reset_default_graph()
						print 'RUN FAIT'
########################################## PLOT ####################################################
########## TEST
if disp_plot == 1:
	figure()
	for i in xrange(len(game_file_test)):
		n_ite_test =  result_game_test[0][i]
		obj_test   = result_game_test[1][i]
		obj_simplex_test = result_simplex_test[i][1]
		n_ite_simplex_test = result_simplex_test[i][0]
		
		rcParams['xtick.labelsize'] = 30
		rcParams['ytick.labelsize'] = 30
	#	axhline(obj_simplex_test,color='r',linewidth=3,label='Obj Func simplex method')
	#	plot(range(len(obj_test)),obj_test,linewidth=3,label='Obj Func DL Agent')
	#	legend(prop={'size': 20})
#		title('GAME TEST:Optimal value',fontsize=33)
		size = len(game_file_test)
		subplot(int(ceil(sqrt(size)))+1,int((sqrt(size))),i+1)
		axhline(n_ite_simplex_test,color='r',linewidth=3,label='N_ite simplex method')
		plot(n_ite_test,'ro',color='b',linewidth=3,label='N_ite DL Agent')
		ylim([0,max(max(n_ite_test),n_ite_simplex_test)+1])
		title('GAME TEST number: '+str(i))
	legend(prop={'size': 20},bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	suptitle('GAMES TEST: Number of iteration achieved by DL_agent and Base case Simplex method per EPOCH',fontsize=33)
	



##########     GAME 1
	figure()
	for i in xrange(len(games_file_train)):
		n_ite,obj,error_batch,grad_weight,sol  = result_games
		n_ite_simplex_1, obj_simplex_1 = result_simplex_train[i]
	
		n_ite_1 = n_ite[i]
		obj_1   = obj[i]
	
		numpy.set_printoptions(threshold='nan')
		print n_ite_1
		#print 'NB OF CONVERGENCE', converge
		### PLOT ###
		rcParams['xtick.labelsize'] = 30
		rcParams['ytick.labelsize'] = 30
		#figure()
		#axhline(obj_simplex_1,color='r',linewidth=3,label='Obj Func simplex method')
		#plot(range(len(obj_1)),obj_1,linewidth=3,label='Obj Func DL Agent')
		#legend(prop={'size': 20})
		#title('GAME 1:Optimal value',fontsize=33)
		size = len(games_file_train)
	        subplot(int(ceil(sqrt(size)))+1,int((sqrt(size))),i+1)
		axhline(n_ite_simplex_1,color='r',linewidth=3,label='N_ite simplex method')
		plot(n_ite_1,'ro',color='b',linewidth=3,label='N_ite DL Agent')
		ylim([0,max(max(n_ite_1),n_ite_simplex_1)+1])
		title('GAME TRAIN number: '+str(i))
	legend(prop={'size': 20},bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	suptitle('GAMES: Number of iteration achieved by DL_agent and Base case Simplex method per EPOCH',fontsize=33)
	
	
	figure()
	plot(range(len(error_batch)),error_batch,linewidth=3)
	title('ALL GAMES: Belman Error accumulated per Replay trained',fontsize=30)
	
	figure()
	plot(range(len(grad_weight)),grad_weight,linewidth=3)
	title('ALL GAMES: Weight Gradient norm per Replay trained',fontsize=30)


	show()

#print 'TEST RESULT: DL AGENT: nb_iteration= ',result_game_test[0][0] ,'obj= ', result_game_test[0][1],' / SIMPLEX HEURISTIC: nb_iteration= ',result_simplex_test[0][0],'obj= ',result_simplex_test[0][1]

