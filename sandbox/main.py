from pylab import *
from Simplex_solver import *
from Simplex import *
import cPickle
import string
import sys

######################################### PARAMS ####################################################
games_file_train = [ 'TSPdata'+str(i)+'.txt' for i in range(1,21)]
game_file_test  = [ 'TSPdata'+str(i)+'.txt' for i in range(21,50)]


i=1
f= open(games_file_train[i],'rb')
x = cPickle.load(f)
f.close()
c = x[0]
A_eq = x[1]
b_eq = x[2]
A_ub = x[3]
b_ub = x[4]

feasible,n_ite_simplex,obj_simplex,sol = TEST(c,A_ub,b_ub,A_eq,b_eq)

if(feasible==0):
	sys.exit()

simplex = SIMPLEX(c,A_ub,b_ub,A_eq,b_eq)
simplex.get_init_tableaux()
simplex.prepare_phase_2()

T = simplex.T

found =1
ite = 0
#HEURISTIC: 'Dantzig' 'Bland' 'Steepest' 

while(found):
	found,pivot=select_pivot(T,'Steepest')
	ite +=1
	print ite
	if(found):
		found,T=play(T,pivot)
		print 'solution found: ', -T[-1,-1]
print 'solution Simplex: ', obj_simplex





