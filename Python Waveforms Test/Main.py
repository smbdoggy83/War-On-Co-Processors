# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:17:10 2022

@author: Seth
"""
from WF_SDK import device       # import instruments
 
"""-----------------------------------------------------------------------"""
 
# connect to the device
device_data = device.open()
 
"""-----------------------------------"""
 
# use instruments here
 
 
"""-----------------------------------"""
 
# close the connection
device.close(device_data)
