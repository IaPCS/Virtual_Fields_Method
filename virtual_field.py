import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def imagesc(x,y,c):
    plt.imshow(c,extent=(x.min(),x.max(),y.min(),y.max()),interpolation='nearest', cmap=cm.gist_rainbow)
    plt.colorbar()
    plt.show()

E = 200000 #MPa=N/mm^2
nu = 0.29

P = 7000 #Applied load in Newton

InputQ11 = E/(1-(nu**2))
InputQ12 = nu*E/(1-nu**2)

# Dimesions of the disk
Rad = 50. #mm
t = 2. #mm

# 2D-DIC parameter
Step = 10.
mmPerPix = 2*Rad/2000 # mm / pixel ratio
SmallArea = (Step*mmPerPix)**2

start = -Rad-(Step*mmPerPix/2)
stepsize = Step*mmPerPix
end = Rad+(Step*mmPerPix/2)
i = 0
x = []
while  start + i * stepsize <= end:
    x.append(start + i * stepsize)
    i+=1
    
xMat = np.zeros((len(x),len(x)),dtype=float)
yMat = np.zeros((len(x),len(x)),dtype=float)

for i in range(0,len(x)):
    xMat[:,i] = x[i]
    yMat[:,i] = x[len(x)-i-1]
    
theta = np.zeros((len(x),len(x)))    
r = np.zeros((len(x),len(x)))

for i in range(0,len(x)):
    rpart = []
    thetapart = []
    for j in range(0,len(x)):
        radius , angle = cart2pol(x[j],x[len(x)-i-1])
        rpart.append(radius)
        thetapart.append(angle)
    theta[i,:] = thetapart
    r[i,:] = rpart

r[r > Rad] = 0


sigma1 = (-2.*P/(np.pi*t))*(np.divide(np.multiply((Rad-yMat),xMat**2),((xMat**2)+(Rad-yMat)**2)**2)+np.divide(np.multiply(Rad+yMat,xMat**2),(xMat**2+(Rad+yMat)**2)**2)-(1./(2.*Rad)))

sigma2 = (-2.*P/(np.pi*t))* (np.divide((Rad-yMat)**3,((xMat**2)+(Rad-yMat)**2)**2) +  np.divide((Rad+yMat)**3,((xMat**2)+(Rad+yMat)**2)**2) - (1./(2.*Rad)))

sigma6 = (2.*P/(np.pi*t)) * ( np.divide(np.multiply((Rad-yMat)**2,xMat),((xMat**2)+(Rad-yMat)**2)**2) - np.divide(np.multiply((Rad+yMat)**2,xMat),((xMat**2)+(Rad+yMat)**2)**2))

epsilon1 = ((InputQ11/((InputQ11**2)-(InputQ12**2)))*sigma1) - ((InputQ12/((InputQ11**2)-(InputQ12**2)))*sigma2)

epsilon2 = -(InputQ12/((InputQ11**2)-(InputQ12**2))*sigma1) + ((InputQ11/((InputQ11**2)-(InputQ12**2)))*sigma2)

epsilon3 = (2./((InputQ11)-(InputQ12)))*sigma6



#print reduced.mask

r[r > 0] = 1

A = np.zeros((2,2))

A[0,0] = np.sum(np.multiply(epsilon2,r))
A[1,1] = np.sum(np.multiply(epsilon2,r))
A[0,1] = np.sum(np.multiply(epsilon1,r))
A[1,0] = np.sum(np.multiply(epsilon1,r))

B = np.zeros((2))

B[0] = (-2.*P*Rad)/(t*SmallArea)
B[1] = 0.

x = np.linalg.solve(A,B)

M = np.zeros((2))
M[0] = InputQ11
M[1] = InputQ12

Qerror = np.divide(x,M)

print Qerror

#imagesc(xMat,yMat,r)
#imagesc(xMat,yMat,theta)
#imagesc(xMat,yMat,t)



    
