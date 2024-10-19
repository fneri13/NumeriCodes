import numpy as np
import matplotlib.pyplot as plt
import os
from NumericalCodes.riemann_problem import RiemannProblem
import pickle


class ShockTube:
    def __init__(self, x, t):
        """
        Initialize the problem with space and time arrays.
        """
        self.xNodes = x
        self.nNodes = len(x)
        self.dx = x[1]-x[0]
        self.timeVec = t
        self.dt = t[1]-t[0]
        self.nTime = len(t)
        self.nInterfaces = self.nNodes+1
        self.gmma = 1.4
        self.GenerateVirtualGeometry()


    def GenerateVirtualGeometry(self):
        """
        Halo-nodes at the boundaries
        """
        self.nNodesHalo = self.nNodes+2
        self.xNodesVirt = np.zeros(self.nNodesHalo)
        self.xNodesVirt[1:-1] = self.xNodes
        self.xNodesVirt[0] = self.xNodes[0]-self.dx
        self.xNodesVirt[-1] = self.xNodes[-1]+self.dx


    def InstantiateSolutionArrays(self):
        """
        Instantiate the containers for the solutions. The first dimension is space, the second is time.
        """
        self.solutionNames = ['Density', 'Velocity', 'Pressure', 'Energy']
        self.solution = {}
        for name in self.solutionNames:
            self.solution[name] = np.zeros((self.nNodesHalo, self.nTime))


    def InstantiateSolutionArraysConservatives(self):
        """
        Instantiate the containers for the solutions. The first dimension is space, the second is time.
        """
        self.solutionConsNames = ['u1', 'u2', 'u3']
        self.solutionCons = {}
        for name in self.solutionConsNames:
            self.solutionCons[name] = np.zeros((self.nNodesHalo, self.nTime))
        


    def InitialConditionsLeftRight(self, dictIn):
        """
        Initialize the conditions based on initial state, defined by right and left values
        """
        dictIn['Energy'] = dictIn['Pressure'] / (self.gmma - 1) / dictIn['Density']
        for name in self.solutionNames:
            self.solution[name][:, 0] = self.CopyInitialState(dictIn[name][0], dictIn[name][1])
    
    def InitialConditionsArrays(self, dictIn):
        """
        Initialize the conditions based on initial state, defined by arrays
        """
        dictIn['Energy'] = dictIn['Pressure'] / (self.gmma - 1) / dictIn['Density']
        for name in self.solutionNames:
            self.solution[name][:, 1:-1] = dictIn[name]


    def CopyInitialState(self, fL, fR):
        """
        Given left and right values, copy these values along the x-axis
        :param fL:
        :param fR:
        :return:
        """
        xmean = 0.5 * (self.xNodesVirt[-1] + self.xNodesVirt[0])
        f = np.zeros_like(self.xNodesVirt)
        for i in range(len(self.xNodesVirt)):
            if self.xNodesVirt[i] <= xmean:
                f[i] = fL
            else:
                f[i] = fR
        return f


    def PlotSolution(self, iTime, folder_name = None, file_name = None):
        """
        Plot the solution at time instant element iTime
        :param iTime: element index in time array
        :param folder_name:
        :param file_name:
        :return:
        """
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0, 0].plot(self.xNodesVirt, self.solution['Density'][:, iTime], '-C0o', ms=2)
        ax[0, 0].set_ylabel(r'Density')

        ax[0, 1].plot(self.xNodesVirt, self.solution['Velocity'][:, iTime], '-C1o', ms=2)
        ax[0, 1].set_ylabel(r'Velocity')

        ax[1, 0].plot(self.xNodesVirt, self.solution['Pressure'][:, iTime], '-C2o', ms=2)
        ax[1, 0].set_ylabel(r'Pressure')

        ax[1, 1].plot(self.xNodesVirt, self.solution['Energy'][:, iTime], '-C3o', ms=2)
        ax[1, 1].set_ylabel(r'Energy')

        fig.suptitle('Time %.3f' % self.timeVec[iTime])

        for row in ax:
            for col in row:
                col.set_xlabel('x')
                col.grid(alpha=.3)

        if file_name is not None and folder_name is not None:
            os.makedirs(folder_name, exist_ok=True)
            plt.savefig(folder_name + '/' + file_name + '.pdf', bbox_inches='tight')
    

    def PlotConsSolution(self, iTime, folder_name = None, file_name = None):
        """
        Plot the solution at time instant element iTime
        :param iTime: element index in time array
        :param folder_name:
        :param file_name:
        :return:
        """
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(self.xNodesVirt, self.solutionCons['u1'][:, iTime], '-C0o', ms=2)
        ax[0].set_ylabel(r'$\rho$')

        ax[1].plot(self.xNodesVirt, self.solutionCons['u2'][:, iTime], '-C1o', ms=2)
        ax[1].set_ylabel(r'$\rho u$')

        ax[2].plot(self.xNodesVirt, self.solutionCons['u3'][:, iTime], '-C2o', ms=2)
        ax[2].set_ylabel(r'$\rho e$')

        fig.suptitle('Time %.3f' % self.timeVec[iTime])

        for col in ax:
            col.set_xlabel('x')
            col.grid(alpha=.3)

        if file_name is not None and folder_name is not None:
            os.makedirs(folder_name, exist_ok=True)
            plt.savefig(folder_name + '/' + file_name + '_conservatives.pdf', bbox_inches='tight')


    def SetBoundaryConditions(self, BCs, it):
        """
        Set the correct boundary condition type
        """
        self.BCtype = BCs
        if BCs=='reflective':
            self.SetReflectiveBoundaryConditions(it)
        else:
            raise ValueError("Unknown boundary condition type")


    def SetReflectiveBoundaryConditions(self, iTime):
        """
        Set the boundary conditions, making use of the halo nodes (index 0 and -1 along axis 0), and then update the conservative variables
        :parameter iTime: time instant index of the simulation
        """
        self.solution['Density'][0, iTime] = self.solution['Density'][1, iTime]
        self.solution['Density'][-1, iTime] = self.solution['Density'][-2, iTime]

        self.solution['Velocity'][0, iTime] = -self.solution['Velocity'][1, iTime]
        self.solution['Velocity'][-1, iTime] = -self.solution['Velocity'][-2, iTime]

        self.solution['Pressure'][0, iTime] = self.solution['Pressure'][1, iTime]
        self.solution['Pressure'][-1, iTime] = self.solution['Pressure'][-2, iTime]

        self.solution['Energy'][0, iTime] = self.solution['Energy'][1, iTime]
        self.solution['Energy'][-1, iTime] = self.solution['Energy'][-2, iTime]

        self.solutionCons['u1'][:, iTime], self.solutionCons['u2'][:, iTime], self.solutionCons['u3'][:, iTime] = (
                    self.GetConsFromPrim(self.solution['Density'][:, iTime], self.solution['Velocity'][:, iTime], self.solution['Pressure'][:, iTime]))


    def SolveSystem(self, flux_method):
        """
        Solve the euler equations using a certain flux_method. Temporal scheme for now is simply forward explicit Euler
        """
        cons = self.solutionCons
        prim = self.solution
        dx, dt = self.dx, self.dt
        for it in range(1, self.nTime):
            print('Time step: %i of %i' %(it, self.nTime))
            for ix in range(1, self.nNodesHalo-1):

                fluxVec_left = self.ComputeFluxVector(ix-1, ix, it-1, flux_method)
                fluxVec_right = self.ComputeFluxVector(ix, ix+1, it-1, flux_method)
                fluxVec_net = fluxVec_left-fluxVec_right

                cons['u1'][ix, it] = cons['u1'][ix, it-1] + dt/dx*fluxVec_net[0]
                cons['u2'][ix, it] = cons['u2'][ix, it-1] + dt/dx*fluxVec_net[1]
                cons['u3'][ix, it] = cons['u3'][ix, it-1] + dt/dx*fluxVec_net[2]

                prim['Density'][ix, it], prim['Velocity'][ix, it], prim['Pressure'][ix, it], prim['Energy'][ix, it] = \
                    self.GetPrimitivesFromCons(cons['u1'][ix, it], cons['u2'][ix, it], cons['u3'][ix, it])
                
            self.SetBoundaryConditions(self.BCtype, it)


    def ComputeFluxVector(self, il, ir, it, flux_method):
        """
        Compute the flux vector given a certain numerical scheme
        """
        if flux_method=='Godunov':
            rhoL = self.solution['Density'][il, it]
            rhoR = self.solution['Density'][ir, it]
            uL = self.solution['Velocity'][il, it]
            uR = self.solution['Velocity'][ir, it]
            pL = self.solution['Pressure'][il, it]
            pR = self.solution['Pressure'][ir, it]
            nx, nt = 51, 51
            x = np.linspace(-self.dx/2, self.dx/2, nx)
            t = np.linspace(0, self.dt, nt)
            riem = RiemannProblem(x, t)
            riem.InitializeState([rhoL, rhoR, uL, uR, pL, pR])
            riem.InitializeSolutionArrays()
            riem.ComputeStarRegion()
            riem.Solve(domain='interface')
            rho, u, p = riem.GetSolutionInTime()
            u1, u2, u3 = self.GetConsFromPrim(rho, u, p)
            u1AVG, u2AVG, u3AVG = np.sum(u1)/len(u1), np.sum(u2)/len(u2), np.sum(u3)/len(u3)
            flux = self.EulerFlux(u1AVG, u2AVG, u3AVG)
            return flux
        else:
            raise ValueError('Unknown flux method')

    def EulerFlux(self, u1, u2, u3):
        """
        Compute Euler equations flux starting from conservative variables. The expressions are given in page 89 Toro book
        """
        flux = np.zeros(3)
        flux[0] = u2
        flux[1] = 0.5*(3-self.gmma)*u2**2/u1+(self.gmma-1)*u3
        flux[2] = self.gmma*u2*u3/u1-0.5*(self.gmma-1)*u2**3/u1**2
        return flux

    def GetPrimitivesFromCons(self, u1, u2, u3):
        """
        Compute primitive variables from conservative
        """
        rho = u1
        u = u2/u1
        e = u3/rho - 0.5*u**2
        p = (self.gmma-1)*rho*e
        return rho, u, p, e


    def GetConsFromPrim(self, rho, u, p):
        """
        Compute conservative variables from primitives
        """
        u1 = rho
        u2 = rho*u
        e = p/(self.gmma-1)/rho
        u3 = rho*(0.5*u**2+e)
        return u1, u2, u3


    def ShowAnimation(self):
        """
        Show animation of the results for all time instants
        """
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for it in range(self.nTime):
            for row in ax:
                for col in row:
                    col.cla()
            ax[0, 0].plot(self.xNodesVirt, self.solution['Density'][:, it], '-C0o', ms=2)
            ax[0, 0].set_ylabel(r'Density')

            ax[0, 1].plot(self.xNodesVirt, self.solution['Velocity'][:, it], '-C1o', ms=2)
            ax[0, 1].set_ylabel(r'Velocity')

            ax[1, 0].plot(self.xNodesVirt, self.solution['Pressure'][:, it], '-C2o', ms=2)
            ax[1, 0].set_ylabel(r'Pressure')

            ax[1, 1].plot(self.xNodesVirt, self.solution['Energy'][:, it], '-C3o', ms=2)
            ax[1, 1].set_ylabel(r'Energy')

            fig.suptitle('Time %.3f' % self.timeVec[it])

            for row in ax:
                for col in row:
                    col.set_xlabel('x')
                    col.grid(alpha=.3)
            plt.pause(1e-3)
    

    def SaveSolution(self, folder_name, file_name):
        """
        Save the full object
        """
        os.makedirs(folder_name, exist_ok=True)
        full_path = folder_name+'/'+file_name+'.pik'
        with open(full_path, 'wb') as file:
            pickle.dump(self, file)
        print('Solution save to ' + full_path + ' !')























