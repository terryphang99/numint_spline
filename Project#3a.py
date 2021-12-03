# MATH 4500 - Final Project: Problem 1
# Terry Phang

from scipy import integrate
from matplotlib import pyplot as plt 
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
import pandas as pd
import site

# Define function for i(t)
# Define T value array
m = 25;
Ta = 0.001;
Tb = 1;
arrT = np.linspace(Ta,Tb,m);
# print(arrT);

# Define integral from 0 to T of i(t)
def i2(t):
    return (10*np.exp(-t/arrT[i])*np.sin(2*np.pi*t/arrT[i]))**2;
# Define Irms function
def IRMS(u,v):
    return np.sqrt((1/u)*v);


# Analytical Solution
asol = [np.sqrt((25/(2*np.exp(1)*(1+4*np.pi**2)))*(8*np.pi**2*np.exp(1)-8*np.pi**2))]*len(arrT);
    
# Method #1: Composite Gaussian Quadrature (n = 5) over each evenly spaced interval
I_RMS1 = []; # Empty RMS array. That is, each element is an integral over i2(t) from 0 to T/2 for all 0.001<T<1
w = [(322-13*np.sqrt(70))/900,
      (322+13*np.sqrt(70))/900,
      128/225,
      (322+13*np.sqrt(70))/900,
      (322-13*np.sqrt(70))/900]; #GQ weights
t = [-1/3*np.sqrt(5+2*np.sqrt(10/7)),
      -1/3*np.sqrt(5-2*np.sqrt(10/7)),
      0,
      1/3*np.sqrt(5-2*np.sqrt(10/7)),
      1/3*np.sqrt(5+2*np.sqrt(10/7))]; # GQ t values
x = [0]*5; # GQ x values
sInt = 1; # Number of subintervals for evaluating integral over 0 to T/2
for i in range(0,len(arrT)): # Loop over all 0.001<T<1
    storage = []; # Create empty storage
    a = 0; # Define lower limit of integral: a = 0
    b = arrT[i]/2; # Define upper limit of integral: b = T/2
    h = (b-a)/sInt; # Split this interval into "sInt" number of subintervals of equal width h (call them Sn's)
    a1 = 0; b1 = 0; # Initialize
    for j in range(0,sInt): # Loop over each Sn of the interval (0,T/2)  
        if a1 < a+(sInt)*h and b1 < a+(sInt+1)*h: # Check to make sure we don't loop over Sn's outside the interval
            a1 = a+j*h; # The lower limit of the jth Sn 
            b1 = a+(j+1)*h; # The upper limit of the jth Sn
        for k in range (0,5): # Find x values over the jth Sn
            x[k] = ((b1-a1)/2)*t[k]+(b1+a1)/2; # Generate x values over the jth Sn
        storage.append(((b1-a1)/2)*(w[0]*i2(x[0])
                              +w[1]*i2(x[1])
                              +w[2]*i2(x[2])
                              +w[3]*i2(x[3])
                              +w[4]*i2(x[4]))); # Append the GQ over jth Sn to "storage"
    IntCGQ = 0; # Initialize a value for integral by Composite Gaussian Quadrature (CGQ)
    for l in range(0,len(storage)): # Add all the values in "storage" to obtain i2(t) by Composite Gaussian Quadrature
        IntCGQ = IntCGQ+storage[l];
    I_RMS1.append(IRMS(arrT[i],IntCGQ)); # Calculate and append root mean square current value to I_RMS array for each T
print("CGQ5 = ", I_RMS1); # Print I_RMS array

errorCGQ = []; # Define an array for the errors;
for i in range(0,len(arrT)): # Calculate and append error of Composite Gaussian Quadrature on each subinterval
    errorCGQ.append(abs(I_RMS1[i]-asol[i]));
print("error = ", errorCGQ); # Print error
    

# Method 2: Composite Closed Newton-Cotes Formulas
n = 4; # n = 1 --> Trapezoidal; n = 2 --> Simpson's; n = 3 --> Simpson's 3/8; n = 4 --> Boole's Rule 
I_RMS2 = [];
sInt = 1;
for i in range(0,len(arrT)): # Loop over all 0.001<T<1
    storage = []; # Create empty storage
    a = 0; # Define lower limit of integral: a = 0
    b = arrT[i]/2; # Define upper limit of integral: b = T/2
    h = (b-a)/sInt; # Split this interval into "sInt" number of subintervals of equal width h (call them Sn's)
    a1 = 0; b1 = 0; # Initialize
    for j in range(0,sInt): # Loop over each Sn of the interval (0,T/2)  
        if a1 < a+(sInt)*h and b1 < a+(sInt+1)*h: # Check to make sure we don't loop over Sn's outside the interval
            a1 = a+j*h; # The lower limit of the jth Sn 
            b1 = a+(j+1)*h; # The upper limit of the jth Sn
            h1 = (b1-a1)/n;
            if n == 1: # Trapezoidal Rule Implementation
                x0 = a1; x1 = x0+h1;
                f0 = i2(x0); f1 = i2(x1);
                Itr = (h1/2)*(f0+f1);
                storage.append(Itr);                
            elif n == 2: # Simpson's Rule implementation
                x0 = a1; x1 = x0+h1; x2 = x0+2*h1;
                f0 = i2(x0); f1 = i2(x1); f2 = i2(x2);
                Isr = (h1/3)*(f0+4*f1+f2);
                storage.append(Isr);
            elif n == 3: # Simpson's 3/8 rule implementation
                x0 = a1; x1 = x0+h1; x2 = x0+2*h1; x3 = x0+3*h1;
                f0 = i2(x0); f1 = i2(x1); f2 = i2(x2); f3 = i2(x3);
                Ister = (3*h1/8)*(f0+3*f1+3*f2+f3);
                storage.append(Ister);
            else: # Boole's Rule implementation
                x0 = a1; x1 = x0+h1; x2 = x0+2*h1; x3 = x0+3*h1; x4 = x0+4*h1;
                f0 = i2(x0); f1=i2(x1); f2=i2(x2); f3=i2(x3); f4=i2(x4);
                Ibr = (2*h1/45)*(7*f0+32*f1+12*f2+32*f3+7*f4);
                storage.append(Ibr);
    IntCCNC = 0; # Initialize value for integral by composite closed Newton-Cotes formulas
    for l in range(0,len(storage)): # Add all the values in "storage" to obtain i2(t) by CCNC
        IntCCNC = IntCCNC+storage[l];
    I_RMS2.append(IRMS(arrT[i],IntCCNC)); # Calculate and append root mean square current value to I_RMS array for each T
print("CCNC"+str(n)+" = ", I_RMS2); # Print I_RMS array

errorCCNC = []; # Define an array for the errors;
for i in range(0,len(arrT)): # Calculate error of CCNC on each subinterval
    errorCCNC.append(abs(I_RMS2[i]-asol[i]));
print("error = ", errorCCNC); # Print Error

# Method #3: Composite Open Newton-Cotes Formulas
n = 3; # n = 0,1,2,3
I_RMS3 = [];
sInt = 1;
for i in range(0,len(arrT)): # Loop over all 0.001<T<1
    storage = []; # Create empty storage
    a = 0; # Define lower limit of integral: a = 0
    b = arrT[i]/2; # Define upper limit of integral: b = T/2
    h = (b-a)/sInt; # Split this interval into "sInt" number of subintervals of equal width h (call them Sn's)
    a1 = 0; b1 = 0; # Initialize
    for j in range(0,sInt): # Loop over each Sn of the interval (0,T/2)  
        if a1 < a+(sInt)*h and b1 < a+(sInt+1)*h: # Check to make sure we don't loop over Sn's outside the interval
            a1 = a+j*h; # The lower limit of the jth Sn 
            b1 = a+(j+1)*h; # The upper limit of the jth Sn
            h1 = (b1-a1)/(n+2);
            if n == 0: # Midpoint Rule
                x0 = a1+h1;
                f0 = i2(x0);
                I0 = 2*h1*f0;
                storage.append(I0);
            elif n == 1: # Open Newton-Cotes n = 1
                x0 = a1+h1; x1 = x0+h1;
                f0 = i2(x0); f1 = i2(x1);
                I1 = (3*h1/2)*(f0+f1);
                storage.append(I1); 
            elif n == 2:# Open Newton-Cotes n = 2
                x0 = a1+h1; x1 = x0+h1; x2 = x0+2*h1;
                f0 = i2(x0); f1 = i2(x1); f2 = i2(x2);
                I2 = (4*h1/3)*(2*f0-f1+2*f2);
                storage.append(I2);
            else: # Open Newton-Cotes n = 3
                x0 = a1+h1; x1 = x0+h1; x2 = x0+2*h1; x3 = x0+3*h1;
                f0 = i2(x0); f1=i2(x1); f2=i2(x2); f3=i2(x3);
                I3 = (5*h1/24)*(11*f0+f1+f2+11*f3);
                storage.append(I3);
    IntCONC = 0;
    for l in range(0,len(storage)):
        IntCONC = IntCONC+storage[l];
    I_RMS3.append(IRMS(arrT[i],IntCONC));
print("CONC"+str(n)+" = ", I_RMS3);

errorCONC = []; # Define an array for the errors;
for i in range(0,len(arrT)): # Calculate error of Composite Gaussian Quadrature on each subinterval
    errorCONC.append(abs(I_RMS3[i]-asol[i]));
print("error = ", errorCONC);
    
# Plot of the numerical results 
plt.figure();
plt.plot(arrT,I_RMS1,'bo');
plt.plot(arrT,I_RMS2,'ro');
plt.plot(arrT,I_RMS3,'go');
plt.plot(arrT, asol, 'k-');
plt.xlim([0,1]);
plt.ylim([3.5,4]);
plt.xlabel('T');
plt.ylabel('I_RMS');
plt.title('Numerical and Analytical I_RMS');
blue_patch = mpatches.Patch(color='b', label='CGQ n = 5');
red_patch = mpatches.Patch(color='r', label='CNC n = 4');
green_patch = mpatches.Patch(color='g',label='ONC n = 3');
black_patch = mpatches.Patch(color='k', label='Analytical Solution')
plt.legend(handles = [blue_patch,red_patch,green_patch,black_patch]);
plt.show();

MyOA = open('Table 3(a).txt','w');
MyOA.write(\
    'Numerical Solutions for Part (a)'+'\n'
    +'T'+'                     '+'CGQ5'+'                  '+'error_CGQ5'+'                    '+'CCNC'
    +'                  '+'error_CCNC'
    +'                    '+'CONC'+'                 '+'error_CONC'
    '\n') 
for i in range(0,len(arrT)):    
    MyOA.write(\
        '\n'
        + '%0f' % arrT[i]\
        + '%26.15f' % I_RMS1[i]    
        + '%26.15E' % errorCGQ[i]\
        + '%26.15f' % I_RMS2[i]\
        + '%26.15E' % errorCCNC[i]\
        + '%26.15f' % I_RMS3[i]\
        + '%26.15E' % errorCONC[i]\
        + '\n')    