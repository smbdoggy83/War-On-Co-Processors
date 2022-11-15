# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:17:10 2022

@author: Seth
"""
#from WF_SDK import device       # import instruments
import time
import ctypes
from WF_SDK import pattern, supplies
from DigiAD2 import DigiAD2 

from sys import platform, path    # this is needed to check the OS type and get the PATH
from os import sep                # OS specific file path separators

# load the dynamic library, get constants path (the path is OS specific)
if platform.startswith("win"):
    # on Windows
    dwf = ctypes.cdll.dwf
    constants_path = "C:" + sep + "Program Files (x86)" + sep + "Digilent" + sep + "WaveFormsSDK" + sep + "samples" + sep + "py"

# import constants
path.append(constants_path)
import dwfconstants as constants
from WF_SDK.device import check_error

"""
An extension of the DigiAD2 class which includes Digital I/O functions
"""
class ExtendedAD2(DigiAD2):
        
    class trigger_source:
        """ trigger source names """
        none = constants.trigsrcNone
        analog = constants.trigsrcDetectorAnalogIn
        digital = constants.trigsrcDetectorDigitalIn
        external = [None, constants.trigsrcExternal1, constants.trigsrcExternal2, constants.trigsrcExternal3, constants.trigsrcExternal4]

    class idle_state:
        """ channel idle states """
        initial = constants.DwfDigitalOutIdleInit
        high = constants.DwfDigitalOutIdleHigh
        low = constants.DwfDigitalOutIdleLow
        high_impedance = constants.DwfDigitalOutIdleZet
    
    
    """-----------------------------------------------------------------------"""

    def generateDig(self, channel, function, frequency, duty_cycle=50, data=[], wait=0, repeat=0, run_time=0, idle=idle_state.initial, trigger_enabled=False, trigger_source=trigger_source.none, trigger_edge_rising=True):
        """
            generate a logic signal
            
            parameters: - channel - the selected DIO line number
                        - function - possible: pulse, custom, random
                        - frequency in Hz
                        - duty cycle in percentage, used only if function = pulse, default is 50%
                        - data list, used only if function = custom, default is empty
                        - wait time in seconds, default is 0 seconds
                        - repeat count, default is infinite (0)
                        - run_time: in seconds, 0=infinite, "auto"=auto
                        - idle - possible: initial, high, low, high_impedance, default = initial
                        - trigger_enabled - include/exclude trigger from repeat cycle
                        - trigger_source - possible: none, analog, digital, external[1-4]
                        - trigger_edge_rising - True means rising, False means falling, None means either, default is rising
        """
        handle = ctypes.c_int(self.handle.value +  1)
        
        # get internal clock frequency
        internal_frequency = ctypes.c_double()
        if dwf.FDwfDigitalOutInternalClockInfo(handle, ctypes.byref(internal_frequency)) == 0:
            check_error()
        
        # get counter value range
        counter_limit = ctypes.c_uint()
        if dwf.FDwfDigitalOutCounterInfo(handle, ctypes.c_int(channel), ctypes.c_int(0), ctypes.byref(counter_limit)) == 0:
            check_error()
        
        # calculate the divider for the given signal frequency
        if function == constants.DwfDigitalOutTypePulse:
            divider = int(-(-(internal_frequency.value / frequency) // counter_limit.value))
        else:
            divider = int(internal_frequency.value / frequency)
        
        # enable the respective channel
        if dwf.FDwfDigitalOutEnableSet(handle, ctypes.c_int(channel), ctypes.c_int(1)) == 0:
            check_error()
        
        # set output type
        if dwf.FDwfDigitalOutTypeSet(handle, ctypes.c_int(channel), function) == 0:
            check_error()
        
        # set frequency
        if dwf.FDwfDigitalOutDividerSet(handle, ctypes.c_int(channel), ctypes.c_int(divider)) == 0:
            check_error()

        # set idle state
        if dwf.FDwfDigitalOutIdleSet(handle, ctypes.c_int(channel), idle) == 0:
            check_error()

        # set PWM signal duty cycle
        if function == constants.DwfDigitalOutTypePulse:
            # calculate counter steps to get the required frequency
            steps = int(round(internal_frequency.value / frequency / divider))
            # calculate steps for low and high parts of the period
            high_steps = int(steps * duty_cycle / 100)
            low_steps = int(steps - high_steps)
            if dwf.FDwfDigitalOutCounterSet(handle, ctypes.c_int(channel), ctypes.c_int(low_steps), ctypes.c_int(high_steps)) == 0:
                check_error()
        
        # load custom signal data
        elif function == constants.DwfDigitalOutTypeCustom:
            # format data
            buffer = (ctypes.c_ubyte * ((len(data) + 7) >> 3))(0)
            for index in range(len(data)):
                if data[index] != 0:
                    buffer[index >> 3] |= 1 << (index & 7)
        
            # load data
            if dwf.FDwfDigitalOutDataSet(handle, ctypes.c_int(channel), ctypes.byref(buffer), ctypes.c_int(len(data))) == 0:
                check_error()
        
        # calculate run length
        if run_time == "auto":
            run_time = len(data) / frequency
        
        # set wait time
        if dwf.FDwfDigitalOutWaitSet(handle, ctypes.c_double(wait)) == 0:
            check_error()
        
        # set repeat count
        if dwf.FDwfDigitalOutRepeatSet(handle, ctypes.c_int(repeat)) == 0:
            check_error()
        
        # set run length
        if dwf.FDwfDigitalOutRunSet(handle, ctypes.c_double(run_time)) == 0:
            check_error()

        # enable triggering
        if dwf.FDwfDigitalOutRepeatTriggerSet(handle, ctypes.c_int(trigger_enabled)) == 0:
            check_error()
        
        if trigger_enabled:
            # set trigger source
            if dwf.FDwfDigitalOutTriggerSourceSet(handle, trigger_source) == 0:
                check_error()
        
            # set trigger slope
            if trigger_edge_rising == True:
                # rising edge
                if dwf.FDwfDigitalOutTriggerSlopeSet(handle, constants.DwfTriggerSlopeRise) == 0:
                    check_error()
            elif trigger_edge_rising == False:
                # falling edge
                if dwf.FDwfDigitalOutTriggerSlopeSet(handle, constants.DwfTriggerSlopeFall) == 0:
                    check_error()
            elif trigger_edge_rising == None:
                # either edge
                if dwf.FDwfDigitalOutTriggerSlopeSet(handle, constants.DwfTriggerSlopeEither) == 0:
                    check_error()

        # start generating the signal
        if dwf.FDwfDigitalOutConfigure(handle, ctypes.c_int(True)) == 0:
            check_error()
        return
    
    def rail_data():
        supplies_data = supplies.data()
        
        supplies_data.master_state = True
        supplies_data.state = True
        supplies_data.voltage = 3.3
        supplies.switch(device_data, supplies_data)
        return

    """-----------------------------------------------------------------------"""

    def close(self):
        """
            reset the instrument
        """
        if dwf.FDwfDigitalOutReset(ctypes.c_int(self.handle.value + 1)) == 0:
            check_error()
        return
    
    def digitalEnable(self, channel):
        """ enables a digital output channel """
        if dwf.FDwfDigitalOutEnableSet(ctypes.c_int(self.handle.value + 1), ctypes.c_int(channel), ctypes.c_int(1)) == 0:
            check_error()
        if dwf.FDwfDigitalOutConfigure(ctypes.c_int(self.handle.value + 1), ctypes.c_int(True)) == 0:
            check_error()
        return

"""-----------------------------------------------------------------------"""
try:
    scope = ExtendedAD2()
    
    
    # connect to the device
    scope.open_scope(20e6, 10e3)
     
    # use instruments here
    #print(scope.read_volt1())
    #scope.get_wav1()
    #scope.set_wav1('square', 1e03, 2, 0, 50, 1, 4, 1)
    #scope.trigger(True, 'analog', 0, 3, edge_rising=False, level=0.25, hysteresis=0.1)
    scope.set_wav1('dc', amplitude=5)
    
    scope.digitalEnable(0)
    
    scope.generateDig(channel=0, 
                     function=pattern.function.pulse, 
                     frequency=0.1)
    #scope.generateDig(channel=2, 
    #                 function=pattern.function.pulse, 
    #                 frequency=2)
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
