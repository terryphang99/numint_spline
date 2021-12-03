# MATH 4500 - Final Project: Problem 2
# Terry Phang

from scipy import *
from matplotlib import pyplot as plt 
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
import pandas as pd
import site

# Define data sets
x = [0,0.375,0.75,1.5,2.25,3,4.5,6,7.5,9,12,15,18,21,24,27,28.5,30];
f = [0,0.801,1.083,1.473,1.74,1.929,2.157,2.25,2.28,
     2.265,2.142,1.923,1.641,1.308,0.924,0.504,0.276,0];
g = [0,-0.369,-0.513,-0.678,-0.783,-0.876,-1.05,-1.191,
     -1.284,-1.338,-1.344,-1.251,-1.101,-0.9,-0.648,-0.369,-0.21,0];
diff = [0]*len(x);
for i in range(0,len(x)):
    diff[i] = f[i]-g[i];

# Perform numerical integration using Spline interpolation

# Initialize and load data arrays
dataX = x;
dataY = diff;

# Initialize n and h array
n = len(dataX)-1;
h = [0]*n;

# Load h array
for j in range(0,n):
    h[j] = dataX[j+1]-dataX[j];

# Initialize and load A matrix
rows, cols = (n+1, n+1) ;
A = [[0 for k in range(cols)] for l in range(rows)]; 

A[0][0] = 1;
A[n][n] = 1;

for j in range (1,n):
    A[j][j-1] = h[j-1];
    A[j][j+1] = h[j];
    A[j][j] = 2*(h[j-1]+h[j]);

print("A = ", A);

# Initialize and load b matrix
vecb = [0]*(n+1);
for j in range(1,n):
     vecb[j] = 3/h[j]*(dataY[j+1]-dataY[j]) - 3/h[j-1]*(dataY[j]-dataY[j-1]);
vecb = np.transpose(vecb);
   
# Solve the linear relationship Ax = b
c = np.float_(np.linalg.solve(A, vecb));

# Obtain a,b,d
a = dataY;
b = [0]*(n+1);
d = [0]*(n+1);
for j in range (0,n):
    b[j] = (1/h[j])*(a[j+1]-a[j]) - (h[j]/3)*(2*c[j]+c[j+1]);
    d[j] = (c[j+1] - c[j])/(3*h[j]);

# Remove last entries of c, transpose a b,c, and d
a = np.delete(a,n);
b = np.delete(b,n);
c = np.delete(c,n);
d = np.delete(d,n);

#print a,b,c,d
print("a = ", a);
print("b = ", b);
print("c = ", c);
print("d = ", d);

MyOB = open('SplineFunctions.txt','w');
MyOB.write(\
    'Natural Cubic Spline Interpolation Functions'+'\n'
    +'Subinterval'+'                              '+'Spline'
    '\n') 
    
# Print Spline Interpolation Function over all intervals
for i in range(0,n):
    xk = dataX[i]
    acoef = a[i];
    bcoef = b[i];
    ccoef = c[i];
    dcoef = d[i];
    print("S_"+str(i)+"(x) = "+str(acoef)+" + "+str(bcoef)+"(x-"+str(xk)+") + "
          +str(ccoef)+"(x-"+str(xk)+")^2 + "+str(dcoef)+"(x-"+str(xk)+")^3");
    MyOB.write('\n'
               + '['
               + '%0.3f' % dataX[i] 
               + ',' 
               + '%0.3f' % dataX[i+1] 
               +']'
               + '                           S_'+str(i)+'(x) = '
               +str(acoef)+' + '+str(bcoef)+'(x-'+str(xk)+') + '
               +str(ccoef)+'(x-'+str(xk)+')^2 + '+str(dcoef)+'(x-'+str(xk)+')^3''\n')
    

# Method 1: Use Composite Gaussian Quadrature with n = 5 to perform integration over sub-subintervals
# Obtain integral between two adjacent data points
# Sum all integrals over full interval [0,30]
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
xCGQ = [0]*5; # GQ x values
sInt = 100; # Number of subintervals for evaluating integral over x_i to x_{i+1}
storage2 = []; # Create empty storage for integral over [0,30]
for i in range(0,len(dataX)-1): # Loop over all dataX
    storage = []; # Create empty storage
    a1 = dataX[i]; # Define lower limit of integral: a1 = x_i
    b1 = dataX[i+1]; # Define upper limit of integral: b1 = x_{i+1}
    h = (b1-a1)/sInt; # Split this subinterval into "sInt" number of sub-subintervals of equal width (call them Sn's)
    a2 = 0; b2 = 0; # Initialize
    xarr = np.linspace(dataX[i],dataX[i+1],10000);
    def y(x):
        return a[i] + b[i]*(x-dataX[i]) + c[i]*(x-dataX[i])**2 + d[i]*(x-dataX[i])**3;
    plt.plot(xarr,y(xarr));
    plt.xlabel('x');
    plt.ylabel('f(x)-g(x)');
    plt.title('Spline Interpolation of diff(x):= f(x)-g(x)');
    for j in range(0,sInt): # Loop over all sub-subintervals   
        if a2 < a1+(sInt)*h and b2 < a1+(sInt+1)*h: # Check to make sure we don't loop over Sn's outside the interval
            a2 = a1+j*h; # The lower limit of the jth Sn 
            b2 = a1+(j+1)*h; # The upper limit of the jth Sn
        for k in range (0,5): # Find x values over the jth Sn
              xCGQ[k] = ((b2-a2)/2)*t[k]+(b2+a2)/2; # Generate x values over the jth Sn;
        storage.append(((b2-a2)/2)*(w[0]*y(xCGQ[0])
                              +w[1]*y(xCGQ[1])
                              +w[2]*y(xCGQ[2])
                              +w[3]*y(xCGQ[3])
                              +w[4]*y(xCGQ[4]))); # Append the GQ over jth Sn to "storage"
    IntCGQ = 0; # Initialize a value for integral by CGQ on a sub-subinterval
    for l in range(0,len(storage)): # Add values in "storage" to obtain integral by CGQ5 on a subinterval
        IntCGQ = IntCGQ+storage[l];
    storage2.append(IntCGQ); # Calculate and append values to an array for all subintervals
TotInt1 = 0; # Initialize
for m in range(0,len(storage2)): # Add all values in this array to obtain total integral over [0,30]
        TotInt1 = TotInt1+storage2[m];
print("CGQ5 = ", TotInt1); # Print array


# Method 2: Use Closed Newton-Cotes formulas to perform integration over sub-subintervals
# Obtain integral between two adjacent data points
# Sum all integrals over full interval [0,30]
storage2 = []; # Define empty storage for integral over [0,30]
n = 1; # n = 1 --> Trapezoidal; n = 2 --> Simpson's; n = 3 --> Simpson's 3/8; n = 4 --> Boole's Rule 
sInt = 100; # Number of subintervals for evaluating integral over x_i to x_{i+1}
for i in range(0,len(dataX)-1): # Loop over all 0.001<T<1
    storage = []; # Create empty storage
    a1 = dataX[i]; # Define lower limit of integral: a = 0
    b1 = dataX[i+1]; # Define upper limit of integral: b = T/2
    h = (b1-a1)/sInt; # Split this interval into "sInt" number of subintervals of equal width h (call them Sn's)
    a2 = 0; b2 = 0; # The first interval of width h
    xarr = np.linspace(dataX[i],dataX[i+1],10000);
    def y(x):
        return a[i] + b[i]*(x-dataX[i]) + c[i]*(x-dataX[i])**2 + d[i]*(x-dataX[i])**3;
    for j in range(0,sInt): # Loop over each Sn of the interval (0,T/2)  
        if a1 < a1+(sInt)*h and b1 < a1+(sInt+1)*h: # Check to make sure we don't loop over Sn's outside the interval
            a2 = a1+j*h; # The lower limit of the jth Sn 
            b2 = a1+(j+1)*h; # The upper limit of the jth Sn
            h = (b2-a2)/n;
            if n == 1: # Trapezoidal Rule Implementation
                x0 = a2; x1 = x0+h;
                f0 = y(x0); f1 = y(x1);
                Itr = (h/2)*(f0+f1);
                storage.append(Itr);                
            elif n == 2: # Simpson's Rule implementation
                x0 = a2; x1 = x0+h; x2 = x0+2*h;
                f0 = y(x0); f1 = y(x1); f2 = y(x2);
                Isr = (h/3)*(f0+4*f1+f2);
                storage.append(Isr);
            elif n == 3: # Simpson's 3/8 rule implementation
                x0 = a2; x1 = x0+h; x2 = x0+2*h; x3 = x0+3*h;
                f0 = y(x0); f1 = y(x1); f2 = y(x2); f3 = y(x3);
                Ister = (3*h/8)*(f0+3*f1+3*f2+f3);
                storage.append(Ister);
            else: # Boole's Rule implementation
                x0 = a2; x1 = x0+h; x2 = x0+2*h; x3 = x0+3*h; x4 = x0+4*h;
                f0 = y(x0); f1=y(x1); f2=y(x2); f3=y(x3); f4=y(x4);
                Ibr = (2*h/45)*(7*f0+32*f1+12*f2+32*f3+7*f4);
                storage.append(Ibr);
    IntCCNC = 0; # Initialize a value for integral by CCNC on a subinterval
    for l in range(0,len(storage)): # Add values in "storage" to obtain integral by CCNC on a subinterval
        IntCCNC = IntCCNC+storage[l];
    storage2.append(IntCCNC); # Calculate and append values to an array for all subintervals
TotInt2 = 0; # Initialize
for m in range(0,len(storage2)): # Add all values in this array to obtain total integral over [0,30]
    TotInt2 = TotInt2+storage2[m];
print("CCNC"+str(n)+" = ", TotInt2); # Print array






















