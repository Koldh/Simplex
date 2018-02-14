from pylab import *
from Simplex_solver import *
from Simplex import *
import cPickle
import string
import sys
import glob
######################################### PARAMS ####################################################
#games_file_train = [ 'TSPdata'+str(i)+'.txt' for i in range(1,21)]
#game_file_test  = [ 'TSPdata'+str(i)+'.txt' for i in range(21,50)]
sys.setrecursionlimit(15000)

def loadgame(gamefile):
	f= open(gamefile,'rb')
	x = cPickle.load(f)
	f.close()
	c = x[0]
	A_eq = x[1]
	b_eq = x[2]
	A_ub = x[3]
	b_ub = x[4]
	feasible,n_ite_simplex,obj_simplex,sol = TEST(c,A_ub,b_ub,A_eq,b_eq)
	return feasible,c,A_ub,b_ub,A_eq,b_eq


#HEURISTIC: 'Dantzig' 'Bland' 'Steepest' 

def search(gamefile):
	feasible,c,A_ub,b_ub,A_eq,b_eq=loadgame(gamefile)
	if(feasible==False):
		return 0
	simplex = SIMPLEX(c,A_ub,b_ub,A_eq,b_eq)
	simplex.get_init_tableaux()
	simplex.prepare_phase_2()
	Ts    = [simplex.T]
	path  = ""
	stacks = []#"0000000000000000000000000000000000000000000000000000000000000000000",[]]
	onestep(Ts,[],path,stacks)
	return clean_paths(stacks)


def clean_paths(stacks):
	#find minimum length
	min_length=200
	for s in stacks:
		if(len(s[0])<min_length):
			min_length = len(s[0])
	good = []
	for s in stacks:
		if(len(s[0])==min_length):
			good.append(s)
	return good



def onestep(T,features,path,stacks,policies_name=['Dantzig','Bland'],policies_keys=["0","1"],current=0):
	print path
	if(len(policies_name)==current):#already tested all paths need to go upper
		if(len(T)>1):#GO HIGHER
			onestep(T[:-1],features[:-1],path[:-1],stacks,policies_name,policies_keys,int(path[-1])+1)
	else:
        	cont,pivot,feat=select_pivot(T[-1],policies_name[current])
        	if(cont==True):#not converged
			cont,t=play(T[-1],pivot)
			if(cont==0):#FAUT TOUT GIVE UP
				onestep(T,features,path,stacks,policies_name,policies_keys,current=current+1)
			else:
        			path+=policies_keys[current]
			        features.append(feat)
				T.append(copy(t))
				onestep(T,features,path,stacks,policies_name,policies_keys,current=0)
        	else:#if converged
			stacks.append([path,features])#register the path #TO OPTIMIZE< ONLY REGISTER IF IT IS THE SMALLEST !
			if(len(T)>1):
				onestep(T[:-1],features[:-1],path[:-1],stacks,policies_name,policies_keys,current=int(path[-1])+1)
			#if other guys to test for this leaf, THIS IS USELESS TO DO AS IT CAN NOT BE SHORTER
			
path  = './tspdata*_upper_bound*.pkl'
files = sort(glob.glob(path))
features = []
for f in files[:1]:
	print f
	features.append(search(f))



