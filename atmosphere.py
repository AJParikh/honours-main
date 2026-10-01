import numpy as np
import math
import matplotlib.pyplot as plt

def atmosphere_lookup(h):
    '''
    This function acts as a lookup table for atmospheric density reference
    values, reference heights, and scale heights for a range of different 
    altitudes from 100 - 1000+ km.  Values from Vallado 4th ed. Table 8-4.
    
    Parameters
    ------
    h : float
        altitude [m]
    
    Returns
    ------
    rho0 : float
        reference density [kg/m^3]
    h0 : float
        reference altitude [m]
    H : float
        scale height [m]

    '''
        
    # Input h in meters, convert to km to read from table
    h = h/1000.
        
    if h <= 100:
        # Assume at this height we have re-entered atmosphere
        rho0 = 0
        h0 = 1
        H = 1
    elif h < 110:
        rho0 = 5.297e-7 * 1e9  # kg/km^3
        h0 = 100.    # km
        H = 5.877    # km    
    elif h < 120:
        rho0 = 9.661e-8 * 1e9  # kg/km^3
        h0 = 110.    # km
        H = 7.263    # km   
    elif h < 130:
        rho0 = 2.438e-8 * 1e9  # kg/km^3
        h0 = 120.    # km
        H = 9.473    # km   
    elif h < 140: 
        rho0 = 8.484e-9 * 1e9  # kg/km^3
        h0 = 130.    # km
        H = 12.636   # km       
    elif h < 150:
        rho0 = 3.845e-9 * 1e9  # kg/km^3
        h0 = 140.    # km
        H = 16.149   # km       
    elif h < 180:
        rho0 = 2.070e-9 * 1e9  # kg/km^3
        h0 = 150.    # km
        H = 22.523   # km       
    elif h < 200:
        rho0 = 5.464e-10 * 1e9  # kg/km^3
        h0 = 180.    # km
        H = 29.740   # km     
    elif h < 250:
        rho0 = 2.789e-10 * 1e9  # kg/km^3
        h0 = 200.    # km
        H = 37.105   # km   
    elif h < 300:
        rho0 = 7.248e-11 * 1e9  # kg/km^3
        h0 = 250.    # km
        H = 45.546   # km       
    elif h < 350:
        rho0 = 2.418e-11 * 1e9  # kg/km^3
        h0 = 300.    # km
        H = 53.628   # km       
    elif h < 400:
        rho0 = 9.518e-12 * 1e9  # kg/km^3
        h0 = 350.    # km
        H = 53.298   # km       
    elif h < 450:
        rho0 = 3.725e-12 * 1e9   # kg/km^3
        h0 = 400.    # km
        H = 58.515   # km     
    elif h < 500:
        rho0 = 1.585e-12 * 1e9   # kg/km^3
        h0 = 450.    # km
        H = 60.828   # km   
    elif h < 600:
        rho0 = 6.967e-13 * 1e9   # kg/km^3
        h0 = 500.    # km
        H = 63.822   # km
    elif h < 700:
        rho0 = 1.454e-13 * 1e9   # kg/km^3
        h0 = 600.    # km
        H = 71.835   # km
    elif h < 800:
        rho0 = 3.614e-14 * 1e9   # kg/km^3
        h0 = 700.    # km
        H = 88.667   # km       
    elif h < 900:
        rho0 = 1.17e-14 * 1e9    # kg/km^3
        h0 = 800.    # km
        H = 124.64   # km       
    elif h < 1000:
        rho0 = 5.245e-15 * 1e9   # kg/km^3
        h0 = 900.    # km
        H = 181.05   # km       
    else:
        rho0 = 3.019e-15 * 1e9   # kg/km^3
        h0 = 1000.   # km
        H = 268.00   # km
    
    
    # Table parameters in km, convert to meters for output
    rho0 *= (1./1e9)
    h0 *= 1000.
    H *= 1000.
    
    
    return rho0, h0, H

hs = np.arange(249000, 251050, 50)
rhos = []
for i, h in enumerate(hs):
    rho0, h0, H = atmosphere_lookup(h)
    rhos.append(rho0*np.exp(-(h-h0)/H))

print(rhos)

plt.plot(hs, rhos)
plt.grid()
plt.ylim([7.08e-11, 7.45e-11])
plt.show()
