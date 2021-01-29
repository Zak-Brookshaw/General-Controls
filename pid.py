import numpy as np


class PID():
    wind_num = 0
    wind = 0
    Kp=1
    toaI=1
    toaD=1
    acc = 0
    pre_error=0
    _dt = 1
    cnt = 0
    def __init__(self, Kp, toaI, toaD):
        
        self.Kp = Kp
        self.toaI = toaI
        self.toaD = toaD
    
    # @property
    # def dt(self):
    #     return self._dt
    
    # @dt.setter
    def dt(self, dt):
        self._dt = dt
    
    def set_wind(self, num:int):
        
        self.wind_num = num
        
        
    def feedback(self, error):
        self.wind+=1
        
        self.acc += error*self._dt 
        der = (error - self.pre_error)/self._dt
        mv_action  = self.Kp*error + self.Kp * self.acc / self.toaI + \
            self.Kp*self.toaD * der
        self.prev_error = error
        
        if self.wind_num and self.wind > self.wind_num:
            self.acc = 0; self.wind = 0
            
        # if self.cnt > 10:
        #     self.acc = 0
            # self.cnt = 0
            
        return mv_action