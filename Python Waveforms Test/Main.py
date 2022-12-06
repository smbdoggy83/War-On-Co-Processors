# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:17:10 2022

@author: Seth
"""
#from WF_SDK import device       # import instruments1
import matplotlib.pyplot as plt
import matplotlib.animation as Animation
import pyformulas as pf
import numpy as np


from time import sleep
import ctypes
from WF_SDK import device, scope, pattern, supplies, static, wavegen
from DigiAD2 import DigiAD2 

from sys import platform, path    # this is needed to check the OS type and get the PATH
from os import sep                # OS specific file path separators

# load the dynamic library, get constants path (the path is OS specific)
if platform.startswith("win"):
    # on Windows
    dwf = ctypes.cdll.dwf
    constants_path = "./"

# import constants
path.append(constants_path)
import dwfconstants as constants
from WF_SDK.device import check_error

"""-----------------------------------------------------------"""
#Initialize the device
device_name = "Analog Discovery 2"

device_data = device.open()
device_data.name = device_name

#Start the positive supply
supplies_data = supplies.data()
supplies_data.master_state = True
supplies_data.state = True
supplies_data.voltage = 5
supplies.switch(device_data, supplies_data)

#Initialize the scope with default settings
scope.open(device_data)

#Set all DIO pins as output
for index in range(16):
    static.set_mode(device_data, index, True)
    


"""-----------------------------------------------------------------------"""

mask = 1;

times = []
buffers = []
fig = plt.figure()

screen = pf.screen(title='Plot')

plt.gcf().canvas.get_renderer()

try:
    
    wavegen.generate(device_data, 
                     channel=1, 
                     function=wavegen.function.dc,
                     offset=5)
    
    count = 0
    
    while True:
        count = count + 1
        
        if count > 20:
            count = 0
            mask ^= 1
        
        static.set_state(device_data, 0, mask)
        
        #Read in data from scope
        buffer, time = scope.record(device_data, channel=1)
        
        # plot
        time = [moment * 1e03 for moment in time]   # convert time to ms
        #times.extend(time)
        times = time
        #buffers.extend(buffer)
        buffers = buffer
        
        fig.clear()
        plt.xlabel("time [ms]")
        plt.ylabel("voltage [V]")
        plt.plot(buffers)
        
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))


        fig.canvas.draw()

        screen.update(image)
        
        
        sleep(0.001)
        


except KeyboardInterrupt: 
    #Stop if Ctrl+C is pressed
    pass

finally:
    screen.close()
    
    # stop the static I/O
    static.close(device_data)
    
    #Close the scope
    scope.close(device_data)
 
    # stop and reset the power supplies
    supplies_data.master_state = False
    supplies.switch(device_data, supplies_data)
    supplies.close(device_data)
 
    """-----------------------------------"""
 
    # close the connection
    device.close(device_data)