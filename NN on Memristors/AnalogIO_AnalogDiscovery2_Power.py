"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision: 12/28/2015

   Requires:                       
       Python 2.7
"""

from ctypes import *
from dwfconstants import *
import time
import sys

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

#declare ctype variables
hdwf = c_int()
sts = c_byte()
IsEnabled = c_bool()
supplyVoltage = c_double()
supplyCurrent = c_double()
supplyPower = c_double()
supplyLoadPercentage = c_double()

#print DWF version
version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
#print ("DWF Version: " + version.value)

#open device
print ("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == hdwfNone.value:
    print ("failed to open device")
    sys.exit()

# set up analog IO channel nodes
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

for i in range(1, 6): # run for 5 seconds i think
  #wait 1 second between readings
  time.sleep(1)
  #fetch analogIO status from device
  dwf.FDwfAnalogIOStatus(hdwf)

  #supply monitor
  dwf.FDwfAnalogIOChannelNodeStatus(hdwf, c_int(3), c_int(0), byref(supplyVoltage))
  dwf.FDwfAnalogIOChannelNodeStatus(hdwf, c_int(3), c_int(1), byref(supplyCurrent))
  supplyPower.value = supplyVoltage.value * supplyCurrent.value
  print ("Total supply power: " + str(supplyPower.value) + "W")

  supplyLoadPercentage.value = 100 * (supplyCurrent.value / 0.2)
  print ("Load: " + str(supplyLoadPercentage.value) + "%")

  # in case of over-current condition the supplies are disabled
  dwf.FDwfAnalogIOEnableStatus(hdwf, byref(IsEnabled))
  if not IsEnabled:
    #re-enable supplies
    print ("Restart")
    dwf.FDwfAnalogIOEnableSet(hdwf, c_int(False)) 
    dwf.FDwfAnalogIOEnableSet(hdwf, c_int(True))

#close the device
dwf.FDwfDeviceClose(hdwf)
