import numpy as np 
from scipy.optimize import OptimizeResult
from pylab import *
"CACA"

class SIMPLEX:
        def __init__(self,c,A_ub,b_ub,learning_mode = 0,A_eq=None,b_eq=None):
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
		self.learning_mode  = learning_mode
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
		nit1, status = self.simplex_phase_1()
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
		

	def simplex_phase_1(self,maxiter=1000,callback='pivot',tol=1.0E-12,nit0=0,bland=False):
		nit = nit0
    		complete = False
    		n_total = self.T.shape[1]-1
        	m = self.T.shape[0]-2
    		if len(self.basis[:m]) == 0:
        		solution = np.zeros(self.T.shape[1] - 1, dtype=np.float64)
    		else:
        		solution = np.zeros(max(self.T.shape[1] - 1, max(self.basis[:m]) + 1),
                            dtype=np.float64)
	#	non_basis = range(n_total)
    	#	for i in xrange(m):
        #		non_basis.remove(self.basis[i])
    		while not complete:
        # Find the pivot column
        		pivcol_found, pivcol = self._pivot_col(tol=1.0E-12, bland=False)
        		if not pivcol_found:
            			pivcol = np.nan
            			pivrow = np.nan
            			status = 0
            			complete = True
       		 	else:
            	# Find the pivot row
           			pivrow_found, pivrow =self._pivot_row(pivcol,phase=1,tol=1.0E-12)
            			if not pivrow_found:
                			status = 3
                			complete = True
					

			if not complete:
            			if nit >= maxiter:
                # Iteration limit exceeded
                			status = 1
                			complete = True
            			else:
                # variable represented by pivcol enters
                # variable in basis[pivrow] leaves
                			self.basis[pivrow] = pivcol
                			pivval = self.T[pivrow][pivcol]
                			self.T[pivrow, :] = self.T[pivrow, :] / pivval
               	  			for irow in range(self.T.shape[0]):
                  				if irow != pivrow:
              				        	self.T[irow, :] = self.T[irow, :] - self.T[pivrow, :]*self.T[irow, pivcol]
                			nit += 1
    		return nit, status





	def prepare_phase_2(self,callback='pivot',tol=1.0E-12, nit0=0, bland=False):
		complete = False
    		n_total = self.T.shape[1]-1
        	m = self.T.shape[0]-1
        # Check if any artificial variables are still in the basis.
        # If yes, check if any coefficients from this row and a column
        # corresponding to one of the non-artificial variable is non-zero.
        # If found, pivot at this term. If not, start phase 2.
        # Do this for all artificial variables in the basis.
        # Ref: "An Introduction to Linear Programming and Game Theory"
        # by Paul R. Thie, Gerard E. Keough, 3rd Ed,
        # Chapter 3.7 Redundant Systems (pag 102)
        	for pivrow in [row for row in range(self.basis.size) if self.basis[row] > self.T.shape[1] - 2]:
            		non_zero_row = [col for col in range(self.T.shape[1] - 1) if self.T[pivrow, col] != 0]
            		if len(non_zero_row) > 0:
                		pivcol = non_zero_row[0]
                # variable represented by pivcol enters
                # variable in basis[pivrow] leaves
                		self.basis[pivrow] = pivcol
                		pivval = self.T[pivrow][pivcol]
                		self.T[pivrow, :] = self.T[pivrow, :] / pivval
                		for irow in range(self.T.shape[0]):
                    			if irow != pivrow:
                        			self.T[irow, :] = self.T[irow, :] - self.T[pivrow, :]*self.T[irow, pivcol]
                	#nit += 1

    		if len(self.basis[:m]) == 0:
        		solution = np.zeros(self.T.shape[1] - 1, dtype=np.float64)
    		else:
        		solution = np.zeros(max(self.T.shape[1] - 1, max(self.basis[:m]) + 1),dtype=np.float64)
                #self.non_basis = range(n_total)
                #for i in xrange(m):
                #       self.non_basis.remove(self.basis[i])
		if self.learning_mode == 0:
        	        ma = np.ma.masked_where(self.T[-1, :-1] >= -tol, self.T[-1, :-1], copy=False)
			self.non_basis = np.squeeze(np.where(ma.mask == False))
#		elif self.learning_mode ==2:
#                	self.non_basis = range(n_total)
#                	for i in xrange(m):
#                		self.non_basis.remove(self.basis[i])
		self.T_non_basis = self.T[:,self.non_basis]
		return solution

	def play(self,n_it,pivcol_action,deter_action,solution,pivcol_found=True,callback='pivot',tol=1.0E-12, nit0=0, bland=False):
	
		m = self.T.shape[0]-1
    		n_total = self.T.shape[1]-1
		if self.learning_mode == 0:
			if deter_action != -1:
				pivcol_action = deter_action
			else:
				pivcol_action = argmax(pivcol_action)
       	        	#ma = np.ma.masked_where(self.T[-1, :-1] >= -tol, self.T[-1, :-1], copy=False)
			#self.non_basis = np.squeeze(np.where(ma.mask == False))
                elif self.learning_mode ==2:
                        self.non_basis = range(n_total)
                        for i in xrange(m):
                                self.non_basis.remove(self.basis[i])
                        pivcol_action = argmax(pivcol_action)
		nit= nit0
                complete = False
		status = 2
		pivcol_found,_ = self._pivot_col()
                if not pivcol_found:
                	pivcol_action = np.nan
                        pivrow = np.nan
                        status = 0
                        complete = True
			reward = 1#-self.T[-1,-1]
			print 'CONVERGED'
			#self.T  = zeros_like(self.T)
			return reward,complete,status
            # Find the pivot row
                pivrow_found, pivrow = self._pivot_row(pivcol_action,2,tol)
                if not pivrow_found:
                	status = 1
                        complete = True
			reward = 0.01#self.T[-1,-1]#0#-n_it
			#self.T  = zeros_like(self.T)
			return reward,complete,status
		if n_it >100:
			status = 1
                        complete = True
                        reward = 0#self.T[-1,-1]#0#-n_it
			return reward,complete,status
                # variable represented by pivcol enters
                # variable in basis[pivrow] leaves
                self.basis[pivrow] = pivcol_action
                pivval = self.T[pivrow][pivcol_action]
                self.T[pivrow, :] = self.T[pivrow, :] / pivval
                for irow in range(self.T.shape[0]):
                    if irow != pivrow:
                        self.T[irow, :] = self.T[irow, :] - self.T[pivrow, :]*self.T[irow, pivcol_action]
                nit += 1
		if complete == False:
			reward =1./(n_it+1)#1./20#n_it#-self.T[-1,pivcol_action]
		if self.learning_mode == 0:
			ma = np.ma.masked_where(self.T[-1, :-1] >= -tol, self.T[-1, :-1], copy=False)
			self.non_basis = np.squeeze(np.where(ma.mask == False))
                elif self.learning_mode ==2:
                        self.non_basis = range(n_total)
                        for i in xrange(m):
                                self.non_basis.remove(self.basis[i])
		#print 'NON BASIS ',self.non_basis, ' ', shape(self.non_basis)
                #self.T_non_basis_next = self.T[:,self.non_basis]
		return reward,complete,status

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
                reduced_cost_feat,steepest_edge_feat,greatest_improv_feat = _pivot_features()
		if heuristic == 'Dantzig':
			pivcol_action = np.ma.where(ma == ma.min())[0][0]
		elif heurisitc == 'Bland':
			pivcol_action = np.ma.where(MASK.mask==False)[0][0]
		elif heuristic == 'Steepest':
			pivcol_action = steepest_edge_feat.argmin()
		elif heuristic == 'Greatest':
			pivcol_action = greatest_improv_feat.argmin()
    		return True, pivcol_action

	def _pivot_features(self):
		#REDUCED COST: c_j
		reduced_cost_feat  = self.T[-1,:-1]
		#STEEPEST EDGE: c_j/norm(T[:,j])
		steepest_edge_feat = self.T[-1,:-1]/norm(T[:-1,:-1],axis=0)
		#GREATEST IMPROVEMENT:  c_j x theta_j
		theta              = self.T[:-1,-1].reshape((-1,1))/self.T[:-1,:-1]
		theta_min          = theta.min(axis=0)
		greatest_improv_feat    = self.T[-1,:-1]*theta_min
	return reduced_cost_feat,steepest_edge_feat,greatest_improv_feat




