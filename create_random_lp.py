from pylab import *
from Simplex_solver import *
import numpy
import cPickle


input_dim = 20
nb_constraint = 10
c  = abs(randn(input_dim))
A = randn(nb_constraint,input_dim)
b = (randn(nb_constraint))
feasible,n_ite_simplex,obj_simplex,sol = TEST(c,A,b)
print sol
print obj_simplex
if feasible==1:
	print 'File created'
	f = open('Randalltest.txt','wb')
	cPickle.dump([c,A,b],f)

