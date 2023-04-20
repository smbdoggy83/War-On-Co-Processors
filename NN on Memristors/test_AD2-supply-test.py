# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:06:12 2023

@author: Seth
"""

from WF_SDK import device, static, supplies       # import instruments
 
from time import sleep                            # needed for delays
 
device_name = "Analog Discovery 2"
 
"""-----------------------------------------------------------------------"""

try: 
        
    # connect to the device
    device_data = device.open()
    device_data.name = device_name
     
    """-----------------------------------"""
     
    # start the positive supply
    supplies_data = supplies.data()
    supplies_data.master_state = True
    supplies_data.state = True
    supplies_data.voltage = 3.3
    supplies.switch(device_data, supplies_data)
    
    # enable positive supply
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(0), c_double(True)) 
    # set voltage to 5 V
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(0), c_int(1), c_double(5.0)) 
    # enable negative supply
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(1), c_int(0), c_double(True)) 
    # set voltage to -5 V
    dwf.FDwfAnalogIOChannelNodeSet(hdwf, c_int(1), c_int(1), c_double(-5.0)) 
    # master enable
    dwf.FDwfAnalogIOEnableSet(hdwf, c_int(True))
    
    # set all pins as output
    for index in range(16):
        static.set_mode(device_data, index, True)
     
    try:
        while True:
            mask = 1
        
    except KeyboardInterrupt:
        # stop if Ctrl+C is pressed
        pass
    
    finally:
        # stop the static I/O
        static.close(device_data)
    
        # stop and reset the power supplies
        supplies_data.master_state = False
        supplies.switch(device_data, supplies_data)
        supplies.close(device_data)
     
        """-----------------------------------"""
     
        # close the connection
        device.close(device_data)

except error as e:
    print(e)
    # close the connection
    device.close(device.data)
