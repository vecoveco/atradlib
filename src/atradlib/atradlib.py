"""Main module."""

import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits.mplot3d import Axes3D
from time import *
import gc

import warnings
warnings.filterwarnings('ignore')

def MCM(H=1000., photons=200,  beta=0.005, ssalbedo=0.95, g=0.85, theta=0.78, phi=0, sim=100000, rnd=42):
    """
    Monte Carlo Method for RTE
    
    # Input:
    # -------
      H : Height of the cloud in m
      photons: number of photons
      beta : volumen extinction coef. in 1/m
      ssalbedo : single scattering albedo [0-1]
      g: asymetrie parameter (1 total forwad scattering)
      theta: radiation incident/elevation angle in rad
      phi: azimuthal radiation direction angle in rad
      sim: Simulations till absorption 
      rnd: random seed
    
    # Output:
    # -------
      coords per photon : [x,y,z] in m
      tran: number of transmitted photons
      refl: number of reflected photons
      abso: number of absorbed photons
    
    """
    
    ## Path length
    def weg(a,rando):
        """
        Path of the photon
        """
        l = -1. * np.log(1-rando) / a # 
        #l=-np.log10(1-np.random.normal(0,0.2,1))*a
        return l

    ## Asymetry Parameter
    def Theta_HG(random,g):
        """
        Asymetry Parameter g
        """
        if g==0:
            Theta=np.arccos(2*random-1);
            return Theta
        else:
            res = (1. + g**2) - ((1.- g**2.)/(1.- g + 2.*g*random))**2
            Theta = np.arccos((1./(2. *g)*res))
            return Theta

    # Rotations Matrix
    def rotmat(theta_old,phi_old):
        """
        Rotation Matrix
        """
        R11 = np.cos(theta_old) * np.cos(phi_old)
        R21 = np.cos(theta_old) * np.sin(phi_old)
        R31 = -np.sin(theta_old)
        R12 = -np.sin(phi_old)
        R22 = np.cos(phi_old)
        R32 = 0
        R13 = np.sin(theta_old) * np.cos(phi_old)
        R23 = np.sin(theta_old) * np.sin(phi_old)
        R33 = np.cos(theta_old)
        A = np.matrix([[R11,R21,R31], [R12,R22,R32], [R13,R23,R33]])
        return np.matrix.transpose(A)
    

    
    # Start für X Y Z 
    x, y, z= 0, 0, H

    # Seed für Zufallszahlengenerator
    rs = np.random.RandomState(rnd)


    # Counts
    tran = 0
    refl = 0
    abso = 0
    

    coords = []

    for jj in range(photons):

        # Theta Sonne
        d = theta 
        # Phi Sonne
        dp = phi

        position = np.zeros([sim+1,3]) 
        position[:] = np.NAN
        position[0,0], position[0,1], position[0,2] = x, y, z
        #print(np.rad2deg(d), np.rad2deg(dp))

        i = 0

        while i < sim:

            #EinPhoton dringt in die Wolke ein ...was passiert?
            l = weg(beta, rs.rand(1))#*H
            
            # Neue Position Local
            position[i+1,0] = position[i,0]+np.sin(d)*np.cos(dp)*l # X-position
            position[i+1,1] = position[i,1]+np.sin(d)*np.sin(dp)*l # Y-Position
            position[i+1,2] = position[i,2]-np.cos(d)*l # Z-Position
            
            # Absorbtion, Transmission, Reflexion oder Streuung?!
            if (position[i+1,2]<=0):
                tran = tran + 1
                break ### Schleife stoppen wenn photon am Boden ankommt 
                #pass

            elif (position[i+1,2]>H):
                refl = refl + 1
                break ### Schleife Stoppen wenn photon wolkenoberkante überschreitet
                #pass

            # Absorbtion mit Wahrscheinlichkeit 1-ssalbedo
            elif (np.random.uniform(0,1,1) > ssalbedo): ### Schleife stoppen wenn photon absorbiert wird
                abso = abso +1
                break
                #pass

            else:
                # Rotations Matrix mit ALTEN THETA und PHI bestimmen
                M = rotmat(theta,phi)

                # Scatterevent Im Einheitssystem (Im Labor):
                # NEUEN THETA und PHI werden berechnet 
                d = Theta_HG(rs.rand(1),g)
                dp = np.random.uniform(0,1,1) * np.pi*2
                #dp = rs.rand(1)* np.pi*2
                                
                # Neue Richtung mit NEUEN THETA und PHI (Im Labor)
                xd_neu = np.sin(d)*np.cos(dp) # X-position
                yd_neu = np.sin(d)*np.sin(dp) # Y-Position
                zd_neu = np.cos(d) # Z-Position
                
                # Rotation der Streuung im Labor mit alten THETA und PHI ins Referenzsystem
                # Rihtung im Referenzsystem = Rotation von Streuevent im Labor
                new_d = np.dot(M,np.array([xd_neu, yd_neu, zd_neu]))

                # Merken der neuen Richtung im Referenzsystem
                # Berechnung von THETA und PHI aus Richtung
                #d = m.acos(new_d[2])
                #dp = m.atan2(new_d[0],new_d[1])
            
                d = np.arccos(new_d[2])
                dp = np.arctan2(new_d[1],new_d[0])
            
            # Lauf-Index erhöhen    
            i = i+1
        coords.append(position)

    gc.collect()    
        
    return coords, tran, refl, abso

