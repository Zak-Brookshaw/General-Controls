from .models import FOPTD
from .pid import PID
import numpy as np
import copy


class TubeCalc(PID, FOPTD):
        
    def __init__(self, tube_radius: float, octagon_length: float, initial_x : list):
        r"""
        
        Simulation of the tube feedback response with the octagon sensor 
        set up

        Parameters
        ----------
        tube_radius : float
            radius of tube in mm
        octagon_l : float
            length of octagon in mm
        initial_x : list
            initial position of tube centers

        Returns
        -------
        None.

        """
        self._R = tube_radius
        self._l = octagon_length/2        
        self._xo = initial_x
        #store stationary positions of sensors
        sqrt2 = np.sqrt(2)
        self._pos = np.array([
            [0, self._l],
            [-self._l/sqrt2, self._l/sqrt2],
            [-self._l, 0],
            [-self._l/sqrt2, -self._l/sqrt2],
            [0, -self._l],
            [self._l/sqrt2, -self._l/sqrt2],
            [self._l, 0],
            [self._l/sqrt2, self._l/sqrt2]
            ])
                                
        
    def set_func(self, *args):
        r"""
        Set the tube response functions

        Parameters
        ----------
        *args : tuple
            tuple of lists of parameters for a FOPTD

        Returns
        -------
        None.

        """
        # initial list to be filled
        self.func = [[None, None], [None, None]]
        # indices of self.func
        l = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for i in range(len(args)):
            A, toa, deadt = args[i]
            row, col = l[i]
            FOPTD.__init__(self, A, toa, deadt)
            self.func[row][col] = copy.deepcopy(self.calc)
            
    def set_dt(self, dt):
        self._dt = dt
        
# %% Controls

    def set_pid(self, *args):
        self._cloop = [None, None]
        
        for i in range(len(args)):
            A, toa, deadt = args[i]
            PID.__init__(self, A, toa, deadt)
            self.dt(self._dt)
            self.set_wind(1000)
            self._cloop[i] = copy.deepcopy(self.feedback)
    
    def set_setpoint(self, setpts:list):
        r"""
        

        Parameters
        ----------
        setpts : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._setpts = setpts
        
    
    def loop(self, predH:int, span:int):
        
        _time = np.arange(0, predH*self._dt, self._dt)
        _func = [
            [self.func[0][0](_time), self.func[0][1](_time)],
            [self.func[1][0](_time), self.func[1][1](_time)]
        ]
        _xc = self._xo
        _yh = np.ones((predH, 2)) * np.array(_xc)
        it = 0
        xc_hist = np.array([self._xo])
        mv_hist = np.zeros((1, 2))


        while it < span:
            
            e1 = self._setpts[0] - _xc[0]
            e2 = self._setpts[1] - _xc[1]
            m1 = self._cloop[0](e1)
            m2 = self._cloop[1](e2)
            
            for i in range(2):
                _temp = _func[i][0]*m1 + _func[i][1]*m2 
                _yh[:, i] = _temp + np.append(_yh[1:, i], _yh[-1, i])
                _xc[i] = _yh[0, i]
                d_real = self._solve_sensor(_xc[i])
                _xc[i] = self.solve_xc(d_real, _xc[i])
            
            it +=1
            xc_hist = np.append(xc_hist, np.array([_xc]), axis=0)
            mv_hist = np.append(mv_hist, np.array([[m1, m2]]), axis=0)
        return xc_hist, mv_hist
            
# %% xc - > distance  
    def _solve_intersect(self, xc):
        r"""
        Solves for the intersection point between Tube and sensor beam

        Parameters
        ----------
        xc : TYPE
            DESCRIPTION.

        Returns
        -------
        x_intersect : TYPE
            DESCRIPTION.

        """
        descrim = np.sqrt(4*xc**2 - 8*(xc**2 - self._R**2))
        x_diag_p = (2*xc + descrim)/4
        x_diag_n = (2*xc - descrim)/4
        
        x_horz_p = self._R + xc
        x_horz_n = -self._R + xc
        
        # catch potential error
        if self._R > xc:
            y_vert = np.sqrt(self._R**2 - xc**2)
        else:
            y_vert = self._l
            
        x_intersect = np.array([
            [0, y_vert],
            [x_diag_n, -x_diag_n],
            [x_horz_n, 0], 
            [x_diag_n, x_diag_n],
            [0, -y_vert],
            [x_diag_p, -x_diag_p], 
            [x_horz_p, 0], 
            [x_diag_p, x_diag_p]
            ])
        
        return x_intersect
    
    
    def _solve_sensor(self, xc):
        r"""
        
        Calculates the distance between the two arrays

        Parameters
        ----------
        sensor_pos : np.array
            DESCRIPTION.
        coords : np.array
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        x_inter = self._solve_intersect(xc)
        vec = self._pos - x_inter
        mags = np.sqrt(np.matmul(vec**2, np.ones((2, 1))))
        
        return mags

# %% distance -> xc

    def _dx_dia(self, xc):
        r"""
        This calculates the x_dia (see OneNote document)
        as well as their derivatives wrt xc

        Parameters
        ----------
        xc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        _dif = 2*self._R**2 - xc**2
        _discrim = np.sqrt(_dif)
        # x_diap = (xc + _discrim)/2
        # x_dian = (xc - _discrim)/2
        
        dx_diap = (1 + xc/_discrim)/2
        dx_dian = (1 - xc/_discrim)/2
        return dx_diap, dx_dian
        # return x_diap, dx_diap, x_dian, dx_dian
          
    def _dsensor(self, xc):
        r"""
        
        Derivative of sensor measurements 

        Parameters
        ----------
        xc : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
           
        dx_diap, dx_dian = self._dx_dia(xc)
        # TODO
        ddiag_n = np.sqrt(2)*dx_dian  # this will need to be changed
        ddiag_p = np.sqrt(2)*dx_diap  # this will need to be changed
        dvert = xc/np.sqrt(self._R**2 - xc**2)
        dhorz_n = 1
        dhorz_p = -1
        
        dsensor = np.array([[dvert], 
                  [ddiag_n], 
                  [dhorz_n],
                  [ddiag_n],
                  [dvert],
                  [ddiag_p],
                  [dhorz_p],
                  [ddiag_p]
                  ])
    
        return dsensor
    
    def solve_xc(self, sensor:np.array, xc_int:float):
        r"""
        

        Parameters
        ----------
        sensor : np.array
            DESCRIPTION.
        xc_int : float
            DESCRIPTION.

        Returns
        -------
        xc : float
            DESCRIPTION.

        """
        
        xc_n = xc_int
        xc_o = xc_int
        er = 1
        num=0
        while er > 10**-6:
            y = self._solve_sensor(xc_o)
            # dy = (self._solve_sensor(xc_o + .01) - self._solve_sensor(xc_o))/.01
            dy = self._dsensor(xc_o)
            eh = sensor - y
            A = dy[:, 0].dot(dy[:, 0])
            B = eh[:, 0].dot(dy[:, 0])
            xc_n = B/A + xc_o
            er = np.abs((xc_n - xc_o)/xc_n)
            xc_o = xc_n
            num+=1
        return xc_n