from pylab import *
from Simplex_solver import *
from Simplex import *
import cPickle
import string
import sys
import glob
import time
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
	print shape(c)
	print shape(A_eq)
	print shape(b_eq)
	print shape(A_ub)
	print shape(b_ub)
	feasible,n_ite_simplex,obj_simplex,sol = TEST(c,A_ub,b_ub,A_eq,b_eq)
	print n_ite_simplex
	print obj_simplex
	return feasible,c,A_ub,b_ub,A_eq,b_eq,n_ite_simplex


#HEURISTIC: 'Dantzig' 'Bland' 'Steepest' 

def search(gamefile):
	feasible,c,A_ub,b_ub,A_eq,b_eq,n_ite_max=loadgame(gamefile)
	if(feasible==False):
		return 0
	simplex = SIMPLEX(c,A_ub,b_ub,A_eq,b_eq)
	simplex.get_init_tableaux()
	simplex.prepare_phase_2()
	Ts    = [simplex.T]
	path  = ""
	stacks = []#"0000000000000000000000000000000000000000000000000000000000000000000",[]]
	k=search2mars(Ts,n_ite_max)
#	onestep(Ts,[],path,stacks)
	return k#clean_paths(stacks)


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





def search2mars(T,max_depth,policies_name=['Dantzig','Bland','Steepest']):
	for i in xrange(max_depth):
		print 'depth: ',i
		cpt,T=twostep(T,policies_name)
		if(T==0):
			return cpt


def twostep(Ts,policies_name):
	newTs =[None]*len(policies_name)*len(Ts)
	cpt=0
	for t in Ts:
		for i in policies_name:
			cont,pivot,features = select_pivot(t,i)
			if cont== True:
				cont,t=play(t,pivot)
				if(cont):
					newTs[cpt]=t
					cpt +=1
			else:
				return cpt,0
	return 0,newTs




#path  = './DATA/tspdata*_upper_bound*10.pkl'
n_cities = int(sys.argv[-1])
path  = './DATA/tspdata*_upper_bound*'+'cities_'+str(n_cities)+'.pkl'

files = sort(glob.glob(path))
features = []
for f in files[2:4]:
	print f
	t = time.time()
	features.append(search(f))
	print time.time()-t


