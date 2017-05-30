# -*- coding: utf-8 -*-
import yaml, os, sys

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
