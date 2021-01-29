import numpy as np


class FOPTD:
    _deadt = 0.0
    _A = 0.0
    _toa = 1.0
    
    def __init__(self, A, toa, deadt):
        
        self._A = A
        self._toa = toa
        self._deadt = deadt
        
    def calc(self, t):
        
        output = self._A*(1 - np.exp(-(t - self._deadt)/self._toa))*(t > self._deadt)
        
        return output
        
