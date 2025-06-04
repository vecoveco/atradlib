#!/usr/bin/env python
# -*- coding: UTF-8 -*-


"""

This is a self made Library for the Lecture :

- Introduction to Atmospheric Radiation and Remote Sensing

- The whole contents is from the Lecture 416 and the Book:

> A First Course In Atmospheric Radiation
> Secon edition
> Grant W. Petty


Creat by V. Pejcic

"""

# Import Librarys
import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits.mplot3d import Axes3D
from time import *
import gc

import warnings
warnings.filterwarnings('ignore')


# Important Global Constants
c = 2.998*10**8         # Speed of light in m/s
h = 6.626*10**-34       # Planck's constant in Js
k = 1.381*10**-23       # Boltzman constant in J/K
sigma = 5.67 *10**-8    # Stefan-Boltzmann in W/ m^2 K^4
solar = 1370            # Solar constant in W/m^2
au = 150*10**11         # Astronomical unit in m
r_earth = 6356.0        # earth radius in km
kw = 2897.0             # Wien's constant in nm K
n_a = 6.022*1e23        # Avogadro Number mole^-1


# Frequenzy
freq_ku_band = 13.6  #GHz, 1e9 in Hz DPR GPM Ku-band
freq_ka_band = 35.5  #GHz, 1e9 in Hz DPR GPM Ka-band
freq_x_band = 9.3    #GHz, 1e9 in Hz BoXPol/JuXPol Uni Bonn
freq_c_band = 5.6    #GHz, 1e9 in Hz DWD Regenradar
freq_k_band = 24.1   #GHz, 1e9 in Hz # MRR in Bonn
freq_s_band = 3.     #GHz, 1e9 in Hz # Kein Bestimmtes S-band


def BeerBougetLambert(F, beta, s):
    """
    Function:
        Bouguer-Lambert-Beer's law describes the attenuation of the
        intensity of a radiation when passing through a medium
        with an absorbent substance, depending on the
        concentration of the absorbent substance and the layer thickness
    Input:
        F       ::: radiation flux density in W/m2 befor absorption
        beta    ::: volumen absorption coeff.
        s       ::: path length
    Output:
        F_att   ::: radiation flux density in W/m2 after absorption
    """
    F_att = F * np.exp(-1* beta * s)
    return F_att


def beta(ni, lam):
    """
    Function:
         Calculation of the volume absorption coefficient
    Input:
        ni    ::: refractive Index
        lam   ::: wellenlaenge
    Output:
        beta  ::: absorptions coeff
    """
    beta = (4* math.pi * ni)/lam
    return beta

def micro2m(wav):
    """
    Function:
        Conversion from micrometre to metre.
    Input:
        wav wave length in micrometres
    Output:
        Wave length in meter
    """
    wav_neu = wav*10**-6
    return wav_neu

def m2micro(wav):
    """
    Function:
        Conversion from metre to micrometre.
    Input:
        wav wave length in meter
    Output:
        wave length in micrometer
    """
    wav_neu = wav*10**6
    return wav_neu

def freq2wav(freq):
    """
    Function: 
        Convert frequency to wavelength
        
    Input:
        Frequenz in per sec/ Herz [Hz] 
        
    Output:
        wave length in meter [m]
    """
    wave = c / freq
    return wave

def wav2freq(wav):
    """
    Funktion: 
        convert wave length in frequency
        
    Input:
        wave length in m
    Output:
        frequency in per sec/in Herz [Hz]
    """
    frequenc = c / wav
    return frequenc

def K2C(Kelvin):
    """
    Funktion:
        Converting Kelvin zu Celsius
    Input:
        Temperature in Kelvin [K]
    Output:
        Temperature in Celsius [C]
    """
    C = Kelvin - 273.15
    return C

def C2K(C):
    """
    Funktion:
        Converting Celsius in Kelvin
    Input:
        Temperatur in Celsius [C]
    Output:
        Temperatur in Kevin [K]
    """
    Kelvin = C + 273.15
    return Kelvin

def planck(wav, T):
    """
    This function calculates the radiation according to Panlk with dependence on
    temperature and wavelength of a black body GP (6.1)
        
    Input:
        wav : Wave length in [µm]
        T : Temperature in [K]
        
    Output:
        B = Radiation in [W/(m^2 sr µm)]
    """
    a1 = (2.0*h*c**2)/(wav**5)
    b1 = (h*c)/(k*T*wav)
    B = a1 * 1/(np.exp(b1)-1.0)
    return B

def emission(intens, emiss):
    """
    Monochromatic emission of a grey body
        
    Input:
        intens : Intensyti in [W/m^2]
        emiss : emission degree in [/]
    Output:
        Intens_neu : Radiated radiance in [W/m^2 sr]
        
    """
    Intens_neu = emiss*intens
    return Intens_neu

def intens2Tb(wav,intens):   
    """

    Calculation of the radiation temperature from the Plank function.
    GP (6.13)
        
    Input:
        wav : wavelength in [µm]
        intens : Intensity in [W/m^2 sr]
    Output:
        Tb : Radiation temperature in [K]
    """
    a1 = (2.0 * h * c**2) / (intens * wav**5)
    b1 = (h * c) / (k * wav)
    Tb = b1 * 1 / (math.log(1.0 + a1))
    return Tb

def stefan_boltzmann(T):
    """
    Stefan-Boltzman_Law Calculates the radiant flux density integrated over
    all wavelengths as a function of the temperature T. GP (6.5)
        
    Input:
        T : Temperature in [K]

    Output:
        Radiation Flux in [W/m^2]
        
    """
    Fbb = sigma * T**4.
    return Fbb

def srtm(asw, alw, A, eps=None):
    """
    Simpel Radiative Models of Atmosphere for the calculation of atmospheres and
    surface temperatures under different emissivities. GP P. 140-143
        
    Input:
        A : Albedo in [/]
        asw : Absorption short wave in [/]
        alw : Absorbtion long wave in [/]
        
    Output:
        Tatmo : atmo. temperature in [/]
        Tsurf : surface temperature in [/]

    """
    if eps == None:
        Sm = S / 4.
        Esurf = ((1. - (1. - asw) * A) * ((2. - asw) / (2. - alw)))
        Tsurf = (Sm / sigma * Esurf)**(1./4.)
        Eatmo = (((1. - A) * (1. - asw) * alw) + (1. + (1. - asw) * A) * asw)\
                / ((2. - alw) * alw)
        Tatmo = (Sm / sigma * Eatmo)**(1./4.)
        return Tatmo, Tsurf
    if eps != None:
        eps = eps
        Sm = S / 4.
        Esurf = ((A - 1.) * (1. - asw) * (1. + (1. - alw)) * (1. - eps)
                 + eps * (A * (1. - asw)**2)-1.) / (eps * (2. - alw))
        Tsurf = (-1. * Sm / sigma * Esurf)**(1. / 4.)

        Eatmo = (((A - 1.) * (1. - asw)) * alw + (1. + ((1. - asw) * A)) * asw)\
                / (alw * (eps * ((1. - alw) + 1.)) + ((1. - alw) * (1. - eps)))

        Tatmo = (-1. *Sm / sigma * Eatmo)**(1. / 4.)
        return Tatmo, Tsurf
    

def rainbow_min_def_ang(m, k):
    """
    Computes a minimum angle of deflection for a Rainbow
    Returns

    Input:
        m : index of refrection in [/]
        k : number if reflections
    Output:
        res : min. Angle der Deflection
    """
    res = np.sqrt((m**2)/(k**2 +2*k))
    res = np.rad2deg(np.arccos(res))

    return res


def optt_cloud(N,H=100, L=0.01, roh=1000, Q=2):
    """
    Function for calculating the optical thickness of a cloud
    
    Input: 
        H : Height of the cloud in [m]
        N : Number of drops in [1/m^3]
        L : Liquid water content in [kg/m^2]
        rho : density in [kg/m^3]
    
    Output: 
        tau : opt. thickness in [/]
    
    """
    ne = 9. * H * np.pi * N * (L**2)  
    za = 16. * (roh**2)
    res = ne/za
    tau = Q * (res)**(1./3.)
    return tau

def transmission(tau, theta):
    """
    
    This function calculates the transmission for a given optical depth
    and zenith angle
    
    Input:
        tau : opt thikness in [/]
        theta : zenith angle in [°]
        
    Output:
        transmission in [/]
    """
    
    res = -1. * (tau / np.cos(np.deg2rad(theta)))
    t = np.exp(res)
    return t
              
    
def swimmingpool(lam, ni, x):
    """
    Function for calculating the transmission in water given
    wave length and imaginary refractive index.
    
    Input:
        lam : wave length in micrometer
        ni : imaginaerteil Refractive index dimensionless
        x : Abstand in mikrometern
    Output:
        t : Distance in micrometres
    """
    ne = -1 * (4 * np.pi * ni * x) 
    ze = lam
    res = ne / ze
    t = np.exp(res)
    return t


def Tb_sat(lam, intens):
    """
    This function calculates the radiation temperature from a given wave length and
     radiation intensity using the inverse
     Planck function.
    
    Input:
        I : Intensity in [W m^-2 µm^-1 sr^-1 ]
            (time 10e6 for including micrometer)
        lam : wave length in [µm]
        
    Output:
        tb :Radiation temperature  
    
    """
    
    a1 = (2.0 * h * c**2.) / (intens * lam**5.) 
    #print (a1)
    b1 = (h * c) / (k * lam)
    #print (b1)
    tb = b1 * 1./np.log(1.0 + a1)
    return tb


def size_para(rr,ww):
    """
    This function calculates the size parameter as a function of
    particle radius to shaft length. GP (12.1)
    
    Input: 
        rr : Particle radius in micrometres [µm]
        ww : wave length in micrometer [µn]
    Output:
        chi : Size Parameter [/]
    """
    chi = (2. * np.pi * rr) / ww
    return chi

def ray_efficiencies(m,chi):
    """
    This function calculates the
     efficiencies from the refractive index and the 
     size parameter using Rayleigh theory. (GP 12.12 - 12.14)

    Input:
        m : Index of refrection in [/]
        chi : Size Parameter in [/]
        
    Output:
        Qe : extinktions Efficiency
        Qs : streuungs Efficiency
        Qa : absorbtions Efficiency
    """
    res = 1. + (chi*+2./15.)*((m**2. -1.) / (m**2. + 2.))*((m**4. + 27.*m**2.+38.)/(2.*m**2. +3.))
    Qe = 4.*chi*(((m**2. -1.) / (m**2. + 2.))*res).imag
    Qs = (8./3.) * chi**4. * abs((m**2. -1.) / (m**2. + 2.))**2.
    Qa =4.*chi*((m**2. -1.) / (m**2. + 2.)).imag 
    
    return Qe, Qs, Qa


def efficiencies(m,chi):
    """
     This function calculates the
     efficiencies with the Mie theory
     from the refractive index and the size parameter.


    Input:
        m : Brechungsindex in [/]
        chi : Size Parameter in [/]

    Output:
        Qe : extinktions Efficiency
        Qs : streuungs Efficiency
        Qa : absorbtions Efficiency
    """
    
    ## calc N
    N=round(chi+4.*chi**(1./3.)+2,0)
    #N=chi+4*chi**(1/3)+2
    
    ## initialize Mie coefficients
    an=np.zeros(int(N), dtype=complex)
    bn=np.zeros(int(N), dtype=complex)
    
    ## initialize efficiencies
    Qext, Qsca, Qabs = 0, 0, 0

    ## initialize recursion terms for Mie coefficients
    W0=complex(np.sin(chi), np.cos(chi))
    Wm1=complex(np.cos(chi),-np.sin(chi))
    A0=1./(np.tan(m*chi))
    ## loop over Mie coefficients and sum contributions to
    ## the efficiencies
    n=0
    while(n<N):
        n = n+1
        An = -n/(m*chi)+1/(n/(m*chi)-A0)
        Wn = ((2*n-1)/chi)*W0-Wm1
        an[n-1] = ((An/m+n/chi)*Wn.real-W0.real)/((An/m+n/chi)*Wn-W0)
        bn[n-1] = ((m*An+n/chi)*Wn.real-W0.real)/((m*An+n/chi)*Wn-W0)
        A0 = An
        Wm1 = W0
        W0 = Wn
        Qext = Qext+2/(chi**2)*(2*n+1)*np.real(an[n-1]+bn[n-1])
        Qsca = Qsca+2/(chi**2)*(2*n+1)*(abs(an[n-1])**2+abs(bn[n-1])**2)

    ## globalize Mie coefficients and efficiencies
    return Qext, Qsca, Qext-Qsca, an, bn


def ray_phase_func(theta):
    """
   This function calculates the Pahsen function from the scattering angle according to
   of the Rayleigh theory. GP (12.10)

    Input:
        theta : scattering angle in [°]

    Output:
        p : Rayleigh Phasen Function in [/]
    """
    p = (3./4.) *(1.+np.cos(np.deg2rad(theta))**2)
    return p



def phase_func(m,chi,mu, nang):
    """
     This function calculates the Pahsen function from the scattering angle according to
     of the Mie theory. (BC)

    Input:
        m : Index of refraction in [/]
        chi : Size parameter in [/]
        mu : Cosinus scattering angle in [/]
        nang : Length Streuwinkelvektors in [/]

    Output:
        p : Mie Phasen Funktion in [/]
    """
    
    N = np.round(chi + 4.*chi**(1./3.) + 2.,0)
    
    Qext, Qsca, Qabs, an, bn = efficiencies(m,chi)

    factor=2./(chi*chi*Qsca)

    S1 = np.zeros(nang, dtype=complex)
    S2 = np.zeros(nang, dtype=complex)
    p11 = np.zeros(nang, dtype=complex)

    i=0

    while(i<(nang)):

        i=i+1
        n=1
        pi0 = 0
        pi1 = 1
        tau1 = mu[i-1]

        while(n<N):
            S1[i-1]=S1[i-1]+(2*n+1)/(n*(n+1))*(an[n-1]*pi1+bn[n-1]*tau1)
            S2[i-1]=S2[i-1]+(2*n+1)/(n*(n+1))*(bn[n-1]*pi1+an[n-1]*tau1)
            n=n+1
            pi1n=(2*n-1)/(n-1)*mu[i-1]*pi1-n/(n-1)*pi0
            tau1=n*mu[i-1]*pi1n-(n+1)*pi1
            pi0=pi1
            pi1=pi1n

        p11[i-1]=factor*(S1[i-1]*np.conj(S1[i-1])+S2[i-1]*np.conj(S2[i-1]))
    
    return p11



def weg(a,rando):
    """
    This function uses the volume extinction coefficient and a
    random number to calculate the path length travelled by a photon in a medium.
    

    Input:
        a : Volumenextinktionskoeffizient in [1/m]
        rando : random number in [/]
        
    Output:
        l : wave length in [m]
        
    """    
    l = -1. * np.log(1-rando) / a # 
    return l


def Theta_HG(random,g):
    """
    This function calculates the random scattering angle according to Henyey Greenstein
    by specifying the asymmetry parameter and a random number. (GP 11.23)
        
    Input:
        random : random nummber in [/]
        g : asymetrieparameter in [/]

    Output:
        Theta : Scattering angle in [rad]
        
    """    
    if g==0:
        Theta=np.arccos(2*random-1);
        return Theta
    else:
        #par=(2. * g * random / (1. - g**2) + 1. / (1. +g ))**2;
        #mu=(1+g**2-1./par)/(2*g);
        #Theta = np.arccos(mu)
        res = (1. + g**2) - ((1.- g**2.)/(1.- g + 2.*g*random))**2
        Theta = np.arccos((1./(2. *g)*res))
        return Theta


def rotmat(theta_old,phi_old):
    """
   This function calculates the rotation matrix of the direction from the
   unit system into the absolute system. (BC)

    Input:
        theta_old : previous elevation angle
        phi_old : previous azimuth angle
        
    Output:
        3x3 Rotationsmatrix
        
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
    


def H_from_OptDic(beta_e, opt_dic):
    """
    Calculation of the height for a given optical thickness and a
    volume extinction coefficient.
    GP (7.32)
        
    Input:
        beta_e : Volumenextinktionscoeffizient in [1/m]
        opt_dic: opt. thickness in [/]
        
    Output:
        H : Layerthickness in [m]
        
    """    

    
    H = opt_dic/beta_e
    
    return H


def gamma(omega_bar, g):
    """
    Calculation of gamma as an intermediate result of the two-stream approximation
    GP (13.25)

    Input:
        omega_bar : single scattering albedo in [/]
        g: Aysmetrieparameter in [/]

    Output:
        gamma : Gamma in [/]

    """
    gamma = 2*np.sqrt(1-omega_bar)*np.sqrt(1-omega_bar*g)
    return gamma

def r_inf(omega_bar, g):
    """
    Calculation of the albedo at the upper edge of the cloud and under the assumption of the
    two-stream approximation for a semi-infinite cloud.
    GP (13.44)

    Input:
        omega_bar : Single scattering albedo in [/]
        g: Aysmetrieparameter in [/]

    Output:
        r_inf

    """
    r_inf = (np.sqrt(1. - omega_bar * g) - np.sqrt(1. - omega_bar)) /\
            (np.sqrt(1. - omega_bar * g) + np.sqrt(1. - omega_bar))
    return r_inf


def tsa_r(g,tau_stern, omega_bar):
    """
    Calculation of the reflection at the upper edge of the cloud and under the assumption of the
     two-stream approximation .
    GP (13.65))

    Input:
        omega_bar : Single scattering albedo in [/]
        g: Aysmetrieparameter in [/]
        tau_stern : opt thickness in [/]

    Output:
        r : Reflexion in [/]

    """
    if omega_bar == 1:
        r = ((1. - g) * tau_stern) / (1. + (1. - g) * tau_stern)
    else:
        n = r_inf(omega_bar, g) * \
            (np.exp(gamma(omega_bar, g) * tau_stern) -
             np.exp(-1. * gamma(omega_bar, g) * tau_stern))

        z = np.exp(gamma(omega_bar, g) *
                   tau_stern) - np.exp(-1. * gamma(omega_bar, g) * tau_stern)\
                                * r_inf(omega_bar, g)**2
        r = n / z
    return r

def tsa_t(g,tau_stern, omega_bar):
    """
    Calculation of the transmission at the upper edge of the cloud and under the assumption of the
    two-stream approximation .
    GP (13.66)

    Input:
        omega_bar : Single scattering albedo in [/]
        g: Aysmetrieparameter in [/]
        tau_stern : opt. thickness in [/]

    Output:
        t : Transmission in [/]

    """
    if omega_bar == 1:
        
        t = (1.) / (1. + (1. - g) * tau_stern)
        
    else:
        n = 1 - r_inf(omega_bar, g)**2
        z = np.exp(gamma(omega_bar, g) * tau_stern) - \
                np.exp(-1. * gamma(omega_bar, g)*tau_stern) * \
                r_inf(omega_bar, g)**2
        t = n / z
    return t

def tsa_tdiff(g,tau_stern, omega_bar):
    """
    Calculation of the direct and diffuse transmission of a cloud under the
    assumption of the two-stream approximation .
    GP (13.69)

    Input:
        omega_bar : Single scattering albedo in [/]
        g: Aysmetrieparameter in [/]
        tau_stern : opt. tchikness in [/]

    Output:
        tdiff : Diffuse transmission in [/]

    """
    if omega_bar == 0:
        tdiff = 0
    if omega_bar == 1:
        tdiff = ((1.) / (1. + (1. - g) * tau_stern)) - np.exp(-tau_stern / 0.5)
    else:
        n = 1-r_inf(omega_bar, g)**2
        z = z = np.exp(gamma(omega_bar, g) * tau_stern) - \
                np.exp(-1. * gamma(omega_bar, g) * tau_stern) *\
                r_inf(omega_bar, g)**2

        tdiff = (n / z) - np.exp(-tau_stern / 0.5)
        
    return tdiff


def MCM(H=1000., photons=200,  beta=0.005, ssalbedo=0.95, g=0.85, theta=0.78, phi=0, sim=100000, rnd=42):
    """
    Monte Carlo Method for RTE
    
    # Input:
    # -------
      H : Height/thikness of the cloud layer in m
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
    

    
    # Start X Y Z 
    x, y, z= 0, 0, H

    # Seed 
    rs = np.random.RandomState(rnd)


    # Counts
    tran = 0
    refl = 0
    abso = 0
    

    coords = []

    for jj in range(photons):

        # Theta sun
        d = theta 
        # Phi sun
        dp = phi

        position = np.zeros([sim+1,3]) 
        position[:] = np.NAN
        position[0,0], position[0,1], position[0,2] = x, y, z

        i = 0

        while i < sim:

            # start photon cloud top
            l = weg(beta, rs.rand(1))#*H
            
            # New Positional Local
            position[i+1,0] = position[i,0]+np.sin(d)*np.cos(dp)*l # X-position
            position[i+1,1] = position[i,1]+np.sin(d)*np.sin(dp)*l # Y-Position
            position[i+1,2] = position[i,2]-np.cos(d)*l # Z-Position
            
            # Absorbtion, Transmission, Reflexion or Scattering
            if (position[i+1,2]<=0):
                tran = tran + 1
                break ### stop if photon at cloud base

            elif (position[i+1,2]>H):
                refl = refl + 1
                break ### stop if photon at cloud top

            # Absorbtion mit Wahrscheinlichkeit 1-ssalbedo
            elif (np.random.uniform(0,1,1) > ssalbedo): ### stop if photon abs
                abso = abso +1
                break

            else:
                # Rotations Matrix with old THETA und PHI 
                M = rotmat(theta,phi)

                # Scatterevent unit system
                # NEW THETA und PHI 
                d = Theta_HG(rs.rand(1),g)
                dp = np.random.uniform(0,1,1) * np.pi*2
                #dp = rs.rand(1)* np.pi*2
                                
                # New Direction with NEW THETA and PHI
                xd_neu = np.sin(d)*np.cos(dp) # X-position
                yd_neu = np.sin(d)*np.sin(dp) # Y-Position
                zd_neu = np.cos(d) # Z-Position
                
                # Rotation of scattering in the unit system with old THETA and PHI in the reference system
                # Rotation in the reference system = rotation of scattering event in the laboratory
                new_d = np.dot(M,np.array([xd_neu, yd_neu, zd_neu]))

                # Calc THETA and PHI from direction
                d = np.arccos(new_d[2])
                dp = np.arctan2(new_d[1],new_d[0])
            
            # Lauf-Index erhöhen    
            i = i+1
        coords.append(position)

    gc.collect()    
        
    return coords, tran, refl, abso


