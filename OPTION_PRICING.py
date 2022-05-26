import numpy as np
import sys
import scipy.linalg as linalg

class AmericanOptions(object):

    def __init__(self, S0, K, r=0.05, T=1,
                sigma=0, D=0, Smax=1, M=1, N=1,
                omega=1, tol=1, is_put=False):
        ###Setup constants
        self.S0= S0
        self.K = K
        self.r = r
        self.T = T
        self.D=D
        self.sigma = sigma 
        self.Smax = Smax
        self.M, self.N = M, N
        self.is_call = not is_put
        self.omega=omega
        self.tol=tol

        ###Setup coordinates on the grid
        self.i_values = np.arange(self.M+1)
        self.j_values = np.arange(self.N+1)
        self.grid = np.zeros(shape=(self.M+1, self.N+1))
        self.boundary_conds = np.linspace(0, Smax, self.M+1)

        #Setup size between points
        self.dS=self.Smax/float(self.M)
        self.dt=self.T/float(self.N)

    def setup_boundary_conditions(self):
        
        #setup of payoffs of exercising the option
        if self.is_call:
            self.payoffs = np.maximum(0,
                self.boundary_conds[1:self.M]-self.K)
        else:
            self.payoffs = np.maximum(0, 
                self.K-self.boundary_conds[1:self.M])
        
        
        #Setup the j+1 values at the boundary
        self.past_values = self.payoffs

    def setup_coefficients(self):
        self.alpha = 0.25*self.dt*(
            (self.sigma**2)*(self.i_values**2)-\
        (self.r-self.D)*self.i_values)
        self.beta= self.dt*0.5*(
            (self.sigma**2)*(self.i_values**2)+self.r)
        self.gamma = 0.25*self.dt*(
            (self.sigma**2)*(self.i_values**2)+
            (self.r-self.D)*self.i_values)
        
        #setup the M2 matrix and the lower and upper triangular matrix of matrix M1
        self.M2 = np.diag(self.alpha[2:self.M], -1) + \
                      np.diag(1-self.beta[1:self.M]) + \
                      np.diag(self.gamma[1:self.M-1], 1)

        self.L=-np.diag(self.alpha[2:self.M], -1) + \
                  np.diag(1+self.beta[1:self.M]) 
        self.U= -np.diag(self.gamma[1:self.M-1],1)


    #function to estimate the price values,
    #using the Gauss_Siedel Method
    def traverse_grid(self):
        #solve the system of non-linear equations
        #using the Guauss-Siedel Method
      
        #setup the j-values
        new_values = np.zeros(len(self.past_values))

        for j in reversed(range(self.N)):
            #compute the R_(j+1)
            rhs = np.dot(self.M2, self.past_values)
            #copy the (j+1)-values
            old_values = np.copy(self.past_values)
            error = sys.float_info.max

            #The Gauss-Siedel method
            while self.tol <error:
                
                new_values=old_values+self.omega*(np.maximum(self.payoffs,linalg.solve(
                    self.L,rhs-np.dot(self.U,old_values)))-old_values)

                error = np.linalg.norm(new_values-old_values)
                old_values = np.copy(new_values)
            
            #update the last column of values
            self.past_values = np.copy(new_values) 
        
        #collect the last column of values
        self.values =np.concatenate(
                ([0], new_values, [0]))

    def interpolate(self):
        #use linear interpolation on final values as 1D array
        return np.interp(self.S0, self.boundary_conds, self.values)


    def price(self):
        self.setup_boundary_conditions()
        self.setup_coefficients()
        self.traverse_grid()
        return self.interpolate()
        

option = AmericanOptions(50,50, r=0.1, T=5./12.,
                sigma=0.4, D=0, Smax=100, M=100, N=84,
                omega=1.2, tol=0.001, is_put=True)

print(option.price())