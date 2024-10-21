import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from numpy import sqrt


class RoeScheme:
    def __init__(self, rhoL, rhoR, uL, uR, pL, pR):
        """
        Initialize the Roe Scheme Numerics.
        """
        self.gmma = 1.4
        self.rhoL = rhoL
        self.rhoR = rhoR
        self.uL = uL
        self.uR = uR
        self.pL = pL
        self.pR = pR
        self.htL = self.ComputeTotalEnthalpy(rhoL, uL, pL)
        self.htR = self.ComputeTotalEnthalpy(rhoR, uR, pR)
        self.u1L, self.u2L, self.u3L = self.GetConsFromPrim(rhoL, uL, pL)
        self.u1R, self.u2R, self.u3R = self.GetConsFromPrim(rhoR, uR, pR)
        self.deltaU1 = self.u1R-self.u1L
        self.deltaU2 = self.u2R-self.u2L
        self.deltaU3 = self.u3R-self.u3L


    def GetConsFromPrim(self, rho, u, p):
        """
        Compute conservative variables from primitives
        """
        u1 = rho
        u2 = rho*u
        e = p/(self.gmma-1)/rho
        u3 = rho*(0.5*u**2+e)
        return u1, u2, u3
    
    def ComputeAveragedVariables(self):
        """
        Compute the roe avg variables for the 1D Euler equations
        """
        self.rhoAVG = sqrt(self.rhoL*self.rhoR)
        self.uAVG = (sqrt(self.rhoL)*self.uL + sqrt(self.rhoR)*self.uR) / (sqrt(self.rhoL)+ sqrt(self.rhoR))
        self.vAVG = 0
        self.wAVG = 0
        self.hAVG = (sqrt(self.rhoL)*self.htL + sqrt(self.rhoR)*self.htR) / (sqrt(self.rhoL)+ sqrt(self.rhoR))
        self.aAVG = sqrt((self.gmma-1)*(self.hAVG-0.5*self.uAVG**2))
    
    def ComputeTotalEnthalpy(self, rho, u, p):
        e = p/(self.gmma-1)/rho
        et = 0.5**u**2 + e
        ht = et+p/rho
        return ht
    
    def ComputeAveragedEigenvalues(self):
        self.lambda_vec = np.array([self.uAVG-self.aAVG, 
                                    self.uAVG,
                                    self.uAVG,
                                    self.uAVG, 
                                    self.uAVG+self.aAVG])
    
    def ComputeAveragedEigenvectors(self):
        self.eigenvector_mat = np.zeros((5, 5))
        
        self.eigenvector_mat[0, 0] = 1
        self.eigenvector_mat[1, 0] = self.uAVG-self.aAVG
        self.eigenvector_mat[4, 0] = self.hAVG-self.uAVG*self.aAVG

        self.eigenvector_mat[0, 1] = 1
        self.eigenvector_mat[1, 1] = self.uAVG
        self.eigenvector_mat[4, 1] = 0.5*self.uAVG**2

        self.eigenvector_mat[2, 2] = 1
        self.eigenvector_mat[3, 3] = 1

        self.eigenvector_mat[0, 4] = 1
        self.eigenvector_mat[1, 4] = self.uAVG+self.aAVG
        self.eigenvector_mat[4, 4] = self.hAVG+self.uAVG*self.aAVG
    
    def ComputeWaveStrengths(self):
        self.alphas = np.zeros(5)
        # self.alphas[1] = (self.gmma-1)/self.aAVG**2*(self.deltaU1*(self.hAVG-self.uAVG**2)+self.uAVG*self.deltaU2-self.deltaU3) 
        # self.alphas[0] = 1/2/self.aAVG * (self.deltaU1*(self.uAVG+self.aAVG)-self.deltaU2-self.aAVG*self.alphas[1])
        # self.alphas[2] = self.deltaU1-(self.alphas[0]+self.alphas[1])
        self.alphas[0] = 1/2/self.aAVG**2 *(self.pR-self.pL-self.rhoAVG*self.aAVG*(self.uR-self.uL))
        self.alphas[1] = self.rhoR-self.rhoL - (self.pR-self.pL)/self.aAVG**2
        self.alphas[2] = self.rhoAVG*self.vAVG
        self.alphas[3] = self.rhoAVG*self.wAVG
        self.alphas[4] = 1/2/self.aAVG**2*(self.pR-self.pL + self.rhoAVG*self.aAVG*(self.uR-self.uL))






    def ComputeFlux(self):
        # let's use the formula 11.27 for the flux

        fluxL = self.EulerFlux(self.u1L, self.u2L, self.u3L)
        fluxR = self.EulerFlux(self.u1R, self.u2R, self.u3R)
        fluxRoe = 0.5*(fluxL+fluxR)
        for iDim in range(5):
            for jVec in range(5):
                fluxRoe[iDim] -= 0.5*self.alphas[jVec]*np.abs(self.lambda_vec[jVec])*self.eigenvector_mat[iDim, jVec]
        
        flux_1D = np.array([fluxRoe[0], fluxRoe[1], fluxRoe[4]])
        return flux_1D
        
    def EulerFlux(self, u1, u2, u3):
        """
        Compute Euler equations flux starting from conservative variables. The expressions are given in page 89 Toro book
        """
        flux = np.zeros(5)
        flux[0] = u2
        flux[1] = 0.5*(3-self.gmma)*u2**2/u1+(self.gmma-1)*u3
        flux[2] = 0
        flux[3] = 0
        flux[4] = self.gmma*u2*u3/u1-0.5*(self.gmma-1)*u2**3/u1**2
        return flux




















