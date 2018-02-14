import numpy as np 
from scipy.optimize import OptimizeResult
from pylab import *
"CACA"
import copy

class SIMPLEX:
        def __init__(self,c,A_ub,b_ub,A_eq=None,b_eq=None):
                self.c = c
                self.A_ub = A_ub
		self.A    = A_ub
		self.b    = b_ub
                self.b_ub = b_ub
		self.A_eq = A_eq
		self.b_eq = b_eq
		self.T = []
		self.T_next = []
		self.n      = len(c)
		self.basis = []
		self.non_basis = [] 
		self.T_non_basis= []
		self.T_non_basis_next = []
        def get_init_tableaux(self):
		# Initialize the tableau to feed to the agent, this step correspond to Simplex phase 1 
		status = 0
    		messages = {0: "Optimization terminated successfully.",
                1: "Iteration limit reached.",
                2: "Optimization failed. Unable to find a feasible"
                   " starting point.",
                3: "Optimization failed. The problem appears to be unbounded.",
                4: "Optimization failed. Singular matrix encountered."}
    		have_floor_variable = False
    		cc = np.asarray(self.c)

         	 # The initial value of the objective function element in the tableau
    		f0 = 0
		bounds =None
		n = self.n

    # Convert the input arguments to arrays (sized to zero if not provided)
		Aeq = np.asarray(self.A_eq) if self.A_eq is not None else np.empty([0, len(cc)])
                Aub = np.asarray(self.A_ub) if self.A_ub is not None else np.empty([0, len(cc)])
                beq = np.ravel(np.asarray(self.b_eq)) if self.b_eq is not None else np.empty([0])
    		bub = np.ravel(np.asarray(self.b_ub)) if self.b_ub is not None else np.empty([0])

    # Analyze the bounds and determine what modifications to be made to
    # the constraints in order to accommodate them.
	        L = np.zeros(self.n, dtype=np.float64)
	        U = np.ones(self.n, dtype=np.float64)*np.inf
    		if bounds is None or len(bounds) == 0:
       			pass
    		elif len(bounds) == 2 and not hasattr(bounds[0], '__len__'):
        # All bounds are the same
        		a = bounds[0] if bounds[0] is not None else -np.inf
        		b = bounds[1] if bounds[1] is not None else np.inf
      		        L = np.asarray(n*[a], dtype=np.float64)
       		        U = np.asarray(n*[b], dtype=np.float64)
  		else:
        		if len(bounds) != self.n:
            			status = -1
            			message = ("Invalid input for linprog with method = 'simplex'.  "
                      "Length of bounds is inconsistent with the length of c")
       		        else:
            			try:
            				for i in range(self.n):
                  		  		if len(bounds[i]) != 2:
                        				raise IndexError()
                    				L[i] = bounds[i][0] if bounds[i][0] is not None else -np.inf
                    				U[i] = bounds[i][1] if bounds[i][1] is not None else np.inf
            			except IndexError:
                			status = -1
					message = ("Invalid input for linprog with "
                           "method = 'simplex'.  bounds must be a n x 2 "
                           "sequence/array where n = len(c).")

    		if np.any(L == -np.inf):
        # If any lower-bound constraint is a free variable
        # add the first column variable as the "floor" variable which
        # accommodates the most negative variable in the problem.
      			self.n = self.n + 1
        		L = np.concatenate([np.array([0]), L])
        		U = np.concatenate([np.array([np.inf]), U])
        		cc = np.concatenate([np.array([0]), cc])
        		Aeq = np.hstack([np.zeros([Aeq.shape[0], 1]), Aeq])
        		Aub = np.hstack([np.zeros([Aub.shape[0], 1]), Aub])
        		have_floor_variable = True

    # Now before we deal with any variables with lower bounds < 0,
    # deal with finite bounds which can be simply added as new constraints.
    # Also validate bounds inputs here.
  		for i in range(n):
        		if(L[i] > U[i]):
            			status = -1
            			message = ("Invalid input for linprog with method = 'simplex'.  "
                       "Lower bound %d is greater than upper bound %d" % (i, i))
		if np.isinf(L[i]) and L[i] > 0:
            		status = -1
            		message = ("Invalid input for linprog with method = 'simplex'.  "
                       "Lower bound may not be +infinity")

        	if np.isinf(U[i]) and U[i] < 0:
            		status = -1
                	message = ("Invalid input for linprog with method = 'simplex'.  "
                       "Upper bound may not be -infinity")

        	if np.isfinite(L[i]) and L[i] > 0:
            # Add a new lower-bound (negative upper-bound) constraint
           		Aub = np.vstack([Aub, np.zeros(n)])
                	Aub[-1, i] = -1
            		bub = np.concatenate([bub, np.array([-L[i]])])
            		L[i] = 0
        	if np.isfinite(U[i]):
            # Add a new upper-bound constraint
                	Aub = np.vstack([Aub, np.zeros(n)])
            		Aub[-1, i] = 1
            		bub = np.concatenate([bub, np.array([U[i]])])
            		U[i] = np.inf
    # Now find negative lower bounds (finite or infinite) which require a
    # change of variables or free variables and handle them appropriately
    		for i in range(0, n):
      			if L[i] < 0:
            			if np.isfinite(L[i]) and L[i] < 0:
                # Add a change of variables for x[i]
                # For each row in the constraint matrices, we take the
                # coefficient from column i in A,
                # and subtract the product of that and L[i] to the RHS b
                			beq = beq - Aeq[:, i] * L[i]
                			bub = bub - Aub[:, i] * L[i]
                # We now have a nonzero initial value for the objective
                # function as well.
                			f0 = f0 - cc[i] * L[i]
            			else:
                # This is an unrestricted variable, let x[i] = u[i] - v[0]
                # where v is the first column in all matrices.
                			Aeq[:, 0] = Aeq[:, 0] - Aeq[:, i]
                			Aub[:, 0] = Aub[:, 0] - Aub[:, i]
                			cc[0] = cc[0] - cc[i]

			if np.isinf(U[i]):
            			if U[i] < 0:
                			status = -1
                			message = ("Invalid input for linprog with "
                           "method = 'simplex'.  Upper bound may not be -inf.")

    # The number of upper bound constraints (rows in A_ub and elements in b_ub)
   		mub = len(bub)
	
    # The number of equality constraints (rows in A_eq and elements in b_eq)
    		meq = len(beq)

    # The total number of constraints
  		m = mub+meq

    # The number of slack variables (one for each of the upper-bound constraints)
  		n_slack = mub

    # The number of artificial variables (one for each lower-bound and equality
    # constraint)
  		n_artificial = meq + np.count_nonzero(bub < 0)
    		try:
                	Aub_rows, Aub_cols = Aub.shape
    		except ValueError:
        	  	raise ValueError("Invalid input.  A_ub must be two-dimensional")
    		try:
        	  	Aeq_rows, Aeq_cols = Aeq.shape
    		except ValueError:
        	  	raise ValueError("Invalid input.  A_eq must be two-dimensional")

    		if Aeq_rows != meq:
        	  	status = -1
                 	message = ("Invalid input for linprog with method = 'simplex'.  "
                   "The number of rows in A_eq must be equal "
                   "to the number of values in b_eq")

    		if Aub_rows != mub:
        	  	status = -1
        	  	message = ("Invalid input for linprog with method = 'simplex'.  "
                   "The number of rows in A_ub must be equal "
                   "to the number of values in b_ub")

    		if Aeq_cols > 0 and Aeq_cols != self.n:
        	  	status = -1
        		message = ("Invalid input for linprog with method = 'simplex'.  "
                   "Number of columns in A_eq must be equal "
                   "to the size of c")

    		if Aub_cols > 0 and Aub_cols != self.n:
        	  	status = -1
        		message = ("Invalid input for linprog with method = 'simplex'.  "
                   "Number of columns in A_ub must be equal to the size of c")
    		if status != 0:
        # Invalid inputs provided
        	  	raise ValueError(message)

    # Create the tableau
    		self.T = np.zeros([m+2, self.n+n_slack+n_artificial+1])

    # Insert objective into tableau
    		self.T[-2, :n] = cc
    		self.T[-2, -1] = f0
    		self.b = self.T[:-2, -1]

    		if meq > 0:
        # Add Aeq to the tableau
        	  	self.T[:meq, :self.n] = Aeq
        # Add beq to the tableau
                  	self.b[:meq] = beq
    		if mub > 0:
        # Add Aub to the tableau
        	  	self.T[meq:meq+mub, :self.n] = Aub
        # At bub to the tableau
        		self.b[meq:meq+mub] = bub
        # Add the slack variables to the tableau
        		np.fill_diagonal(self.T[meq:m, self.n:self.n+n_slack], 1)

    # Further set up the tableau.
    # If a row corresponds to an equality constraint or a negative b (a lower
    # bound constraint), then an artificial variable is added for that row.
    # Also, if b is negative, first flip the signs in that constraint.
    		slcount = 0
    		avcount = 0
    		self.basis = np.zeros(m, dtype=int)
    		r_artificial = np.zeros(n_artificial, dtype=int)
    		for i in range(m):
        		if i < meq or self.b[i] < 0:
            # basic variable i is in column n+n_slack+avcount
            			self.basis[i] = self.n+n_slack+avcount
            			r_artificial[avcount] = i
            			avcount += 1
            			if self.b[i] < 0:
                			self.b[i] *= -1
                			self.T[i, :-1] *= -1
            			self.T[i, self.basis[i]] = 1
            			self.T[-1, self.basis[i]] = 1
        		else:
            # basic variable i is in column n+slcount
            			self.basis[i] = self.n+slcount
            			slcount += 1

    # Make the artificial variables basic feasible variables by subtracting
    # each row with an artificial variable from the Phase 1 objective
    		for r in r_artificial:
        	  	self.T[-1, :] = self.T[-1, :] - self.T[r, :]
		status = self.simplex_phase_1()
		tol = 1.0E-12
    # if pseudo objective is zero, remove the last row from the tableau and
    # proceed to phase 2
    		if abs(self.T[-1, -1]) < tol:
        # Remove the pseudo-objective row from the tableau
        	  	self.T = self.T[:-1, :]
        # Remove the artificial variable columns from the tableau
        		self.T = np.delete(self.T, np.s_[self.n+n_slack:self.n+n_slack+n_artificial], 1)
    		else:
        # Failure to find a feasible starting point
        	  	status = 2

    		if status != 0:
        	  	message = messages[status]
        	  	if disp:
            			print(message)
        		return OptimizeResult(x=np.nan, fun=-self.T[-1, -1], nit=nit1, status=status,
                      message=message, success=False)
		

	def simplex_phase_1(self,maxiter=1000,callback='pivot',tol=1.0E-12,nit0=0,heuristic='Dantzig'):
    		complete = False
		nit      = nit0
    		n_total  = self.T.shape[1]-1
        	m        = self.T.shape[0]-2
    		if len(self.basis[:m]) == 0:
        		solution = np.zeros(self.T.shape[1] - 1, dtype=np.float64)
    		else:
        		solution = np.zeros(max(self.T.shape[1] - 1, max(self.basis[:m]) + 1),
                            dtype=np.float64)
    		while not complete:
        		pivcol_found, pivcol = self._pivot_col(tol=1.0E-12, heuristic=heuristic)
        		if not pivcol_found:
            			pivcol = np.nan
            			pivrow = np.nan
            			status = 0
            			complete = True
       		 	else:
           			pivrow_found, pivrow =self._pivot_row(pivcol,phase=1,tol=1.0E-12)
            			if not pivrow_found:
                			status = 3
                			complete = True
					

			if not complete:
            			if nit >= maxiter:
                			status = 1
                			complete = True
            			else:
                			self.basis[pivrow] = pivcol
                			pivval = self.T[pivrow][pivcol]
                			self.T[pivrow, :] = self.T[pivrow, :] / pivval
               	  			for irow in range(self.T.shape[0]):
                  				if irow != pivrow:
              				        	self.T[irow, :] = self.T[irow, :] - self.T[pivrow, :]*self.T[irow, pivcol]
					nit+=1
    		return status



	def prepare_phase_2(self,callback='pivot',tol=1.0E-12, nit0=0, bland=False):
		complete = False
    		n_total = self.T.shape[1]-1
        	m = self.T.shape[0]-1
        	for pivrow in [row for row in range(self.basis.size) if self.basis[row] > self.T.shape[1] - 2]:
            		non_zero_row = [col for col in range(self.T.shape[1] - 1) if self.T[pivrow, col] != 0]
            		if len(non_zero_row) > 0:
                		pivcol = non_zero_row[0]
                		self.basis[pivrow] = pivcol
                		pivval = self.T[pivrow][pivcol]
                		self.T[pivrow, :] = self.T[pivrow, :] / pivval
                		for irow in range(self.T.shape[0]):
                    			if irow != pivrow:
                        			self.T[irow, :] = self.T[irow, :] - self.T[pivrow, :]*self.T[irow, pivcol]
        def _pivot_row(self, pivcol,phase, tol=1.0E-12):
                if phase == 1:
                        k = 2
                else:
                        k = 1
                ma = np.ma.masked_where(self.T[:-k, int(pivcol)] <= tol, self.T[:-k, int(pivcol)], copy=False)
                if ma.count() == 0:
                        return False, np.nan
                mb = np.ma.masked_where(self.T[:-k, pivcol] <= tol, self.T[:-k, -1], copy=False)
                q = mb / ma
                return True, np.ma.where(q == q.min())[0][0]
	def _pivot_col(self,tol=1.0E-12, heuristic='Dantzig'):
                ma = np.ma.masked_where(self.T[-1, :-1] >= -tol, self.T[-1, :-1], copy=False)
                if ma.count() == 0:
                        return False, np.nan
                steepest_edge_feat,greatest_improv_feat = pivot_features(self.T)
                if heuristic == 'Dantzig':
                        pivcol_action = np.ma.where(ma == ma.min())[0][0]
                elif heurisitc == 'Bland':
                        pivcol_action = np.ma.where(MASK.mask==False)[0][0]
                elif heuristic == 'Steepest':
                        pivcol_action = steepest_edge_feat.argmin()
                elif heuristic == 'Greatest':
                        pivcol_action = greatest_improv_feat.argmin()
                return True, pivcol_action



####################### AGENT SELECTION
def play(T,pivcol_action,tol=1.0E-12):
                #pivcol_action is an integer
	newT = copy.deepcopy(T)
	pivrow_found, pivrow = pivot_row(newT,pivcol_action,tol)
        if not pivrow_found:
                return False,T
        pivval             = newT[pivrow][pivcol_action]
        newT[pivrow, :]  = newT[pivrow, :] / pivval
        for irow in range(T.shape[0]):
        	if irow != pivrow:
                	newT[irow, :] = newT[irow, :] - newT[pivrow, :]*newT[irow, pivcol_action]
        return True,newT


def pivot_row(T, pivcol,tol=1.0E-12):
        k = 1
        ma = np.ma.masked_where(T[:-k, int(pivcol)] <= tol, T[:-k, int(pivcol)], copy=False)
        if ma.count() == 0:
                return False,0
        mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
        q = mb / ma
        return True, np.ma.where(q == q.min())[0][0]


def pivot_features(T):
	#STEAPEST
        steepest_edge_feat   = T[-1,:-1]/norm(T[:-1,:-1],axis=0)
	#GREATEST
        mask                 = T[:-1,:-1]<0
        renorm               = T[:-1,:-1]-T[:-1,:-1]*mask
        theta                = T[:-1,-1].reshape((-1,1))/renorm
        theta_min            = argmin(theta,axis=0)
        greatest_improv_feat = T[-1,:-1]*theta_min
        return steepest_edge_feat,greatest_improv_feat


def select_pivot(T,heuristic,tol=1.0E-12):
	ma = T[-1, :-1] < -tol #true if negative
        if ma.sum() == 0:#safe check
	        return False,-1,0
        steepest_edge_feat,greatest_improv_feat = pivot_features(T)
        if heuristic == 'Dantzig':
	        pivcol_action = argmin(T[-1,:-1])
        elif heuristic == 'Bland':
        	pivcol_action = find(ma)[0]
        elif heuristic == 'Steepest':
                pivcol_action = steepest_edge_feat.argmin()
        elif heuristic == 'Greatest':
                pivcol_action = greatest_improv_feat.argmin()
	return True,pivcol_action,asarray([T[-1,:-1], steepest_edge_feat,greatest_improv_feat])




