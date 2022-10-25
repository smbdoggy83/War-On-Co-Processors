# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:54:04 2022

@author: Seth
"""

# from ctypes import * # old
import ctypes # good

class data:
    """
        stores the device handle and the device name
    """
    handle = ctypes.c_int(0)
    name = ""
 
def open():
    """
        open the first available device
    """
    # this is the device handle - it will be used by all functions to "address" the connected device
    device_handle = ctypes.c_int()
    # connect to the first available device
    dwf.FDwfDeviceOpen(ctypes.c_int(-1), ctypes.byref(device_handle))
    data.handle = device_handle
    return data