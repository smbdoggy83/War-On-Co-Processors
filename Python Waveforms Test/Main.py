# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:17:10 2022

@author: Seth
"""
#from WF_SDK import device       # import instruments
import DigiAD2 
scope = DigiAD2()

"""-----------------------------------------------------------------------"""
 
# connect to the device
scope.open_scope()
 
"""-----------------------------------"""
 
# use instruments here
 
 
"""-----------------------------------"""
 
# close the connection
scope.close()
