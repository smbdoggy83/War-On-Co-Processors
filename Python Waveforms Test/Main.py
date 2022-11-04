# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:17:10 2022

@author: Seth
"""
#from WF_SDK import device       # import instruments
import time
import ctypes
from WF_SDK import pattern
from DigiAD2 import DigiAD2 

"""
An extension of the DigiAD2 class which includes Digital I/O functions
"""
class ExtendedAD2(DigiAD2):
    class Data:
        handle = ctypes.c_int()
        name = ''
    
    def getDeviceData(self):
        data = self.Data()
        data.handle = self.handle
        data.name = 'Digilent Analog Discovery 2'
        return data

"""-----------------------------------------------------------------------"""
try:
    scope = ExtendedAD2(1)
    
    # connect to the device
    scope.open_scope(20e6, 10e3)
     
    """-----------------------------------"""
    
    data = scope.getDeviceData()
     
    # use instruments here
    #print(scope.read_volt1())
    #scope.get_wav1()
    #scope.set_wav1('square', 1e03, 2, 0, 50, 1, 4, 1)
    #scope.trigger(True, 'analog', 0, 3, edge_rising=False, level=0.25, hysteresis=0.1)
    scope.set_wav1('square', amplitude=5, frequency=1)
    pattern.generate(device_data=data,
                     channel=1, 
                     function=pattern.function.pulse, 
                     frequency=1)
    pattern.generate(device_data=data, 
                     channel=2, 
                     functional=pattern.function.pulse, 
                     frequency=2)
    time.sleep(5)
    #scope.close_wav1()

    #scope.get_wav1()

except Exception as e: 
    """-----------------------------------"""

    # close the connection
    scope.close_wav1()
    scope.close_scope()
    scope.close()
    print('Cleanup Complete')
    
    raise e
    
 
# close the connection
scope.close_wav1()
scope.close_scope()
scope.close()
