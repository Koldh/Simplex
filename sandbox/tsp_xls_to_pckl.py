import cPickle
from pylab import *
import pandas as pd

for i in xrange(1,51):
	csv = pd.read_csv("tspdata"+str(i)+".csv",sep=',',header=None)
	Data = csv.values
	print Data
	c     = Data[0]
	A_eq  = Data[1:11]
	b_eq  = Data[11]
	A_ub  = Data[12:24]
	b_ub  = Data[-1]

	# DELETE NAN
	index = argwhere(isnan(b_eq))
	b_eq = delete(b_eq,index)
	index = argwhere(isnan(b_ub))
	b_ub = delete(b_ub,index)

	print c
	print A_eq
	print b_eq
	print A_ub
	print b_ub
	f = open('TSPdata'+str(i)+'.txt','wb')
	cPickle.dump([c,A_eq,b_eq,A_ub,b_ub],f)
	f.close()
