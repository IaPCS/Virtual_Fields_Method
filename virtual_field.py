import numpy as np
import numpy.ma as ma
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import *


def imagesc(x,y,c):
    plt.imshow(c,extent=(x.min(),x.max(),y.min(),y.max()),interpolation='nearest', cmap=cm.gist_rainbow)
    plt.colorbar()
    plt.show()

deck = Deck("./deck.yaml")
coordinates = Coordinates( deck )

sigma1 = (-2.*P/(np.pi*t))*(np.divide( np.multiply((Rad-yMat),xMat**2),((xMat**2)+(Rad-yMat)**2)**2 ) + np.divide(np.multiply(Rad+yMat,xMat**2),(xMat**2+(Rad+yMat)**2)**2))

#print ((xMat**2)+(Rad-yMat)**2)**2

print sigma1[0,0]
#imagesc(xMat,yMat,reduced)
#imagesc(xMat,yMat,theta)
#imagesc(xMat,yMat,t)



    
