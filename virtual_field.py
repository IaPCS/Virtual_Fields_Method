import numpy as np
import numpy.ma as ma
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
    
reduced = ma.masked_where(r > Rad,r)



sigma1 = (-2.*P/(np.pi*t))*(np.divide(np.multiply((Rad-yMat),xMat**2),((xMat**2)+(Rad-yMat)**2)**2)+np.divide(np.multiply(Rad+yMat,xMat**2),(xMat**2+(Rad+yMat)**2)**2))

#print ((xMat**2)+(Rad-yMat)**2)**2

print sigma1[0,0]
#imagesc(xMat,yMat,reduced)
#imagesc(xMat,yMat,theta)
#imagesc(xMat,yMat,t)



    
