import cPickle
from pylab import *
import pandas as pd
import glob


def transformer(DATA,n):
	csv  = pd.read_csv(DATA,sep=',',header=None)
	Data = csv.values
	c    = Data[0,:-1]
	A_eq = Data[1:2*n+1,:-1]
	b_eq = Data[2*n+1]
	A_ub = Data[2*n+2:2*n+2+(n-1)*(n-2),:-1]
	b_ub = Data[-1]

        index = argwhere(isnan(b_eq))
        b_eq = delete(b_eq,index)
        index = argwhere(isnan(b_ub))
        b_ub = delete(b_ub,index)
        f = open(DATA[:-4]+'.pkl','wb')
        cPickle.dump([c,A_eq,b_eq,A_ub,b_ub],f)
        f.close()

data_names = sort(glob.glob('./tspdata*_upper_bound4_cities_5.dat'))
for i in data_names:
	transformer(i,5)




