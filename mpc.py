import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from .utilities import Utilities


class MPC(Utilities):
    
    def __init__(self, predH:int, manH:int, numCV:int, numMV:int, t_step):
        r"""
        This class creates a model predictive controller, uses scipy minimize 
        scalar function to minimize the SSE of the resulting controller

        Parameters
        ----------
        predH : int
            DESCRIPTION.
        manH : int
            DESCRIPTION.
        numCV : int
            DESCRIPTION.
        numMV : int
            DESCRIPTION.
        t_step : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.predH = predH; self.manH = manH; self.numCV = numCV
        self.numMV = numMV; self.t_step = t_step
        # control function list in shape of numCV X numMV
        self._control_actions = [[np.zeros(predH) for j in range(numMV)] 
                                 for i in range(numCV)]
        
        self._prevdu = np.zeros((numMV*manH, 1))
        self._yh = np.zeros((predH*numCV, 1))
        self._u = np.zeros((numMV, 1))
        
        self.index_mv = np.array([i*manH for i in range(numMV)])
        self.index_cv = np.array([i*predH for i in range(numCV)])
        
    def set_initial_MV(self, uo:np.array):
        r"""
        

        Parameters
        ----------
        uo : np.array
            Initial position of manipulated variables

        Returns
        -------
        None.

        """
        uo = self._shape_check(uo, (self.numMV, 1), dtype=float)
        self._u = uo
        
    
    def set_setpoint(self, setpt:float):
        r"""
        sets the setpoint for each control variable

        Parameters
        ----------
        setpt : array-like
            setpoints

        Returns
        -------
        None.

        """
        
        if len(setpt) == self.numCV and self.numCV > 1:
            self.setpoint = np.array([s*np.ones(self.predH) for s in setpt])
            self.setpoint = self.setpoint.reshape((self.numCV*self.predH, 1))
            
        elif isinstance(setpt, (float, int)):
            self.setpoint = np.array(setpt*np.ones((self.predH, 1)))
            
        else:
            raise ValueError("Bad Setpoint")
            
    
    def set_saturation(self, lb:np.array, ub:np.array):
        r"""
        Sets linear constraints on optimizer to not allow for MV saturation to
        be exceeded

        Parameters
        ----------
        lb : np.array
            lower limit
        ub : np.array
            upper limit

        Returns
        -------
        None.

        """
        lb = self._shape_check(lb, (self.numMV, 1))
        ub = self._shape_check(ub, (self.numMV, 1))
        #building transform matrix
        one = np.ones(self.manH)
        A = np.zeros((self.manH * self.numMV, self.numMV))
        trl_ones = np.tril(np.ones((self.manH, self.manH)))
        constrain_mat = np.zeros((self.numMV*self.manH, self.numMV*self.manH))
        
        for i in range(self.numMV):
            _rowL = i*self.manH
            _rowH = _rowL + self.manH
            A[_rowL:_rowH, i] = one
            constrain_mat[_rowL:_rowH, _rowL:_rowH] = trl_ones
            
            
        _dlb = lb - self._u
        _dub = ub - self._u
        self._lowerbound = np.matmul(A, _dlb)
        self._upperbound = np.matmul(A, _dub)       
        self._form_bound = A
        self._constrain_mat = constrain_mat
        
    
    def set_mv_weights(self, weights:np.array):
        r"""
        

        Parameters
        ----------
        weights : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        weights = self._shape_check(weights, (self.numMV, 1))
        
        _temp = np.matmul(self._form_bound, weights)
        W = np.eye(self.manH*self.numMV) * _temp
        self._W = W
        
        
        
    
    def attach_func(self, func, CV:int, MV:int):
        r"""
        function to set the control action functions, ie how a change in MV j
        will effect CV i
        
        These functions MUST produce an array with shape (n,)

        Parameters
        ----------
        func : TYPE
            mathematical representation of the control action
        CV : int
            Control variable index
        MV : int
            Manipulated variable index

        Returns
        -------
        None.

        """
        self._control_actions[CV-1][MV-1] = func
        
      
    def linear_control_mat(self):
        r"""
        creates the matrix of the 

        Returns
        -------
        None.

        """
        #store common array lengths
        _row_length = self.numCV*self.predH
        _col_length = self.numMV*self.manH
        
        #create shift matrix
        self.H = np.eye(self.predH, k=-1)
        self.H[self.predH-1, self.predH-1] = 1
        #initialize control action mantrix
        self._lin_X = np.zeros((_row_length, _col_length))
        time = np.arange(0, self.predH*self.t_step, self.t_step)
        
        #create control action mantrix
        for i in range(self.numCV):
            #row indexes
            row_indL = i*self.predH
            row_indH = row_indL + self.predH
            
            for j in range(self.numMV):
                #temporary matrix for storage
                _temp = self._control_actions[i][j](time)[:, np.newaxis]
                
                for k in range(self.manH):
                    #column index
                    _col_ind = j*self.manH + k
                    self._lin_X[row_indL:row_indH, _col_ind] = _temp[:, 0]
                    #create the next _temp array to be stored next
                    _temp = np.matmul(self.H, _temp)
        
        self._lin_Xstep = self._lin_X[:, np.array([i*self.manH for i in range(self.numMV)])]
    
    def _sse(self, du):
        r"""
        calculates the sum of square errors
        
        Parameters
        ----------
        du : np.array
            DESCRIPTION.

        Returns
        -------
        None.

        """
        du = self._shape_check(du, (self.numMV*self.manH , 1))
        e = self.setpoint - self._yh - np.matmul(self._lin_X, du)
        w = np.matmul(self._W, du)
        
        
        return np.matmul(e.T, e)[0] + np.matmul(w.T, w)[0]
        
    def _jac_sse(self, du):
        r"""
        calculates the jacobian of the sum of square errors
        
        Parameters
        ----------
        du : np.array
            DESCRIPTION.

        Returns
        -------
        None.

        """
        du = self._shape_check(du, (self.numMV*self.manH , 1))
        eh = self.setpoint - self._yh
        XTX = np.matmul(self._lin_X.T, self._lin_X)
        WTW = np.matmul(self._W.T, self._W)
        
        
        de = -2*np.matmul(self._lin_X.T, eh) + 2*np.matmul(XTX, du) + 2*np.matmul(WTW, du)
        
        
        return de[:, 0]
        
        
    def linear_compute(self):
        r"""
        
        Calculates the optimal manipulated variable path over the manipulation
        Horizon        
        
        Raises
        ------
        ValueError
            If no solution found

        Returns
        -------
        du
            optimal manipulated variable change

        """
        Constraints = LinearConstraint(self._constrain_mat, 
                                       self._lowerbound[:, 0], 
                                       self._upperbound[:, 0])
        sol = minimize(self._sse, self._prevdu, jac=self._jac_sse, constraints=(Constraints))
        
        if sol.success:
            self._prevdu = sol.x
            return sol.x
            
        else:
            raise ValueError("No solution found")
        
    def update_yh(self, du):
        r"""
        Update the manipulated variable change to the 
        
        Parameters
        ----------
        du : np.array
            an (numMV, 1) shape array of the manipulated variable changes enacted
        
        Returns
        -------
        None.

        """
        du = self._shape_check(du, (self.numMV, 1))
        
        for i in range(self.numCV):
            _rowL = i*self.predH
            _rowH = (i+1)*self.predH
            self._yh[_rowL:_rowH, 0] = np.append(self._yh[_rowL+1:_rowH], self._yh[_rowH-1])
            
        self._yh += np.matmul(self._lin_Xstep, du) 
        self._u += du
        expanded_du = np.matmul(self._form_bound, du)
        self._lowerbound+=-expanded_du
        self._upperbound+=-expanded_du
    
    def get_yh(self):
        r"""
        Get current value of all control variables

        Returns
        -------
        None.

        """
        
        
        return self._yh[self.index_cv, 0]
        