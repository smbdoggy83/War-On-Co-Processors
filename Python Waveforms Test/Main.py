# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:17:10 2022

@author: Seth
"""
#from WF_SDK import device       # import instruments
from DigiAD2 import DigiAD2 
scope = DigiAD2(1)

"""-----------------------------------------------------------------------"""
 
# connect to the device
#scope.close_scope()
#scope.__init__()
scope.open_scope(20e6, 10e3)
 
"""-----------------------------------"""
 
# use instruments here
#print(scope.read_volt1())
#scope.get_wav1()
#scope.set_wav1('square', 1e03, 2, 0, 50, 1, 4, 1)
scope.trigger(True, 'analog', 0, 3, edge_rising=False, level=0.25, hysteresis=0.1)
scope.set_wav1('square')
#scope.close_wav1()
#scope.cool_PS()
#scope.get_wav1()

 
"""-----------------------------------"""
 
# close the connection
scope.close_wav1()
scope.close_scope()
scope.close()
