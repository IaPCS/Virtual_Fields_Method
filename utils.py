# -*- coding: utf-8 -*-
import yaml, os, sys
import numpy as np
import numpy.ma as ma

class Deck():

    def __init__(self, inputFile):
        if not os.path.exists(inputFile):
            print "Error: Could not find " + inputFile
            sys.exit(1)
        else:
            with open(inputFile,'r') as f:
                self.doc = yaml.load(f)
                
            self.read_yaml()


    def check_presence(self, category, parameter):
        if not parameter in self.doc[category]:
            print parameter + "?"
            sys.exit(1)
        else:
            return self.doc[category][parameter]        
                
                
    def read_yaml(self):
        yamlfile = self.doc
        print "Input data:", yamlfile
        
        if not "Material" in yamlfile:
            print "Material?"
            sys.exit(1)
        else:
            self.E = self.check_presence("Material", "E") 
            self.nu = self.check_presence("Material", "nu")    
            self.Q11 = self.E / (1-(self.nu**2))
            self.Q12 = self.nu*self.E/(1-self.nu**2)
              
        if not "Geometry" in yamlfile:
            print "Geometry?"
            sys.exit(1)
        else:
            self.Radius = self.check_presence("Geometry", "Radius") 
            self.Thickness = self.check_presence("Geometry", "Thickness") 
            
        if not "DIC" in yamlfile:
            print "DIC?"
            sys.exit(1)
        else:
            self.Step = self.check_presence("DIC", "Step")  
            self.mmPerPix = self.check_presence("DIC", "mmperpixel")
            self.SmallArea = (self.Step*self.mmPerPix)**2
            

class Coordinates():
    
    def __init__(self, deck):
        self.xycoordinates(deck)
        self.polarcoordinates()
    
    def xycoordinates(self, deck):
        start = -deck.Radius-(deck.Step*deck.mmPerPix/2)
        stepsize = deck.Step*deck.mmPerPix
        end = deck.Rad+(deck.Step*deck.mmPerPix/2)  
        x = []
        while  start + i * stepsize <= end:
            x.append(start + i * stepsize)
            i+=1
        self.x = x   
        
        xMat = np.zeros((len(x),len(x)),dtype=float)
        yMat = np.zeros((len(x),len(x)),dtype=float)  
        
        for i in range(0,len(x)):
            xMat[:,i] = x[i]
            yMat[:,i] = x[len(x)-i-1]
        self.xMat = xMat
        self.yMat = yMat
        
    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
        
    def polarcoordinates(self):
        theta = np.zeros((len(self.x),len(self.x)))    
        r = np.zeros((len(self.x),len(self.x)))

        for i in range(0,len(self.x)):
            rpart = []
            thetapart = []
            for j in range(0,len(self.x)):
                radius , angle = self.cart2pol(self.x[j],self.x[len(self.x)-i-1])
                rpart.append(radius)
                thetapart.append(angle)
            theta[i,:] = thetapart
            r[i,:] = rpart
            
        r[r > Rad] = 0
        self.theta = theta
        self.r = r
