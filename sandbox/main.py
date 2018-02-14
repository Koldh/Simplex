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

def loadgame(gamefile,testing_rules='Dantzig'):
	f= open(gamefile,'rb')
	x = cPickle.load(f)
	f.close()
	c = x[0]
	A_eq = x[1]
	b_eq = x[2]
	A_ub = x[3]
	b_ub = x[4]
	feasible,n_ite_simplex,obj_simplex,sol = TEST(c=c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,rules=testing_rules)
	print 'n_iteration simplex :',n_ite_simplex, 'rules: ', testing_rules
	print obj_simplex
	return feasible,c,A_ub,b_ub,A_eq,b_eq,n_ite_simplex


#HEURISTIC: 'Dantzig' 'Bland' 'Steepest' 'Greatest' 

def search(gamefile,testing_rules):
	feasible,c,A_ub,b_ub,A_eq,b_eq,n_ite_max=loadgame(gamefile,testing_rules)
	if(feasible==False):
		return 0
	simplex = SIMPLEX(c,A_ub,b_ub,A_eq,b_eq)
	simplex.get_init_tableaux()
	simplex.prepare_phase_2()
	Ts    = [simplex.T]
	p,f=search2mars(Ts,n_ite_max,[])
	f = asarray(f)
	print f.shape, p
	return p,f#clean_paths(stacks)


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



def baseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])

def path2arr(path):
	return array([int(p) for p in path])

def rr(path,b):
	return sum(path2arr(path)*b**(arange(len(path))[::-1]))

def search2mars(T,max_depth,features=[],policies_name=['Dantzig','Bland','Steepest','Greatest']):
	for i in xrange(max_depth):
		print 'depth: ',i
		cpt,T,features=twostep(T,features,policies_name)
		if(T==0):
			path = baseN(cpt,len(policies_name))
			if(len(path)<i):
				path='0'*(i-len(path))+path
#			print path,len(features),[len(features[k]) for k in xrange(len(features))],[rr(path[:k],len(policies_name)) for k in xrange(1,len(features)+1)]
			return path,[features[k-1][rr(path[:k],len(policies_name))] for k in xrange(1,len(features)+1)]


def twostep(Ts,features,policies_name):
	newTs = [None]*len(policies_name)*len(Ts)
	feats = [None]*len(policies_name)*len(Ts)
	cpt   = 0
	for t in Ts:
		for i in policies_name:
			cont,pivot,localfeats = select_pivot(t,i)
			if cont== True:
				cont,t=play(t,pivot)
				if(cont):
					newTs[cpt]=t
					feats[cpt]=localfeats
					cpt      +=1
			else:
#				features.append(feats)
				path = baseN(cpt,len(policies_name))[:-1]
				cpt  = rr(path,len(policies_name))
				return cpt,0,features
	features.append(feats)
	return 0,newTs,features




#path  = './DATA/tspdata*_upper_bound*10.pkl'
n_cities = int(sys.argv[-1])
path  = './DATA/tspdata*_upper_bound*'+'cities_'+str(n_cities)+'.pkl'
testing_rules = 'Steepest'
files = sort(glob.glob(path))
data = []
for f in files:
	print f
	t = time.time()
	data.append(search(f,testing_rules))
	print time.time()-t

k = open('DATA_TSP.pkl','wb')
cPickle.dump(data,k)
k.close()




