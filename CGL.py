# -*- coding: utf-8 -*-
"""
@author: William
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from matplotlib import animation

def main():
    #The main parameters
    N = 2**8
    dt = 0.0001
    Nt = 200
    a_1 = 0
    a_2 = 0
    b_1 = 0.001 #dissipative, heat like, term
    b_2 = 1 #dispersive, schrodinger like, term
    c_1 = 1
    c_2 = 1
    #The inital data
    u_0 = random_initial_data(N,2.5)
    
    #Animating the solution. See the Animate class below.   
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-3, 3))
    line1, = ax.plot([], [], lw=2, label = "Re u")
    line2, = ax.plot([], [], lw=2, label = "Im u")
    plt.legend(loc=1)
    
    animate = Animate(line1, line2, dt, Nt, u_0, a_1, a_2, b_1, b_2, c_1, c_2)
    anim = animation.FuncAnimation(fig, animate, 
                                   frames=1000, interval=20)
    
    anim.save('CGL.mp4', fps=30)
    
def nonlinear_CGL_solution(t, Nt, u_0, a_1, a_2, b_1, b_2, c_1, c_2):
    '''Solvers the, nonlinear, complex Ginzburg-Landau equation
    $$\partial_tu = (a_1+ia_2)u + (b_1+ib_2)\Delta u -(c_1+ic_2)|u|^2u$$
    using a operaotr splitting algorithm, in particular strang splitting.
    '''
    dt = 1.0*t/Nt
    u = u_0
    for i in range(0,Nt):
        u = linear_CGL_solution(dt/2, u, a_1, a_2, b_1, b_2)
        u = nonlinearity_solver(dt, u, c_1, c_2)
        u = linear_CGL_solution(dt/2, u, a_1, a_2, b_1, b_2)
    
    return u     

def linear_CGL_solution(t, u_0, a_1=1, a_2=1, b_1=1, b_2=1):
    '''Solves the linearised complex Ginzburg-Landau equation,
    $$\partial_tu = (a_1+ia_2)u + (b_1+ib_2)\Delta u $$
    using the fast Fourier transform.
    '''
    N = len(u_0)
    k = np.arange(-N/2, N/2)+0j
    Fu_0 = fft.fft(u_0)
    Fu_0 = fft.fftshift(Fu_0)
    Fu = np.exp((a_1 + np.complex(0,1)*a_2 - (b_1 + np.complex(0,1)*b_2)*4*np.pi**2*k**2)*t)*Fu_0
    Fu = fft.ifftshift(Fu)
    u = fft.ifft(Fu)
    
    return u

def nonlinearity_solver(t,u_0, c_1, c_2):
    '''An auxillry function to solve $$\partial_t u = |u|^2u$$ with $$u(0)=u_0$$ 
    for very small t. There is an exact solution to this equation so an obvious
    improvement would be to code this exact solution. I do not believe this 
    would result in a large increase in accuracy however.
    '''
    u = u_0 - (c_1 + np.complex(0,1)*c_2)*t*np.abs(u_0)**2*u_0
    return u
    
def random_initial_data(N,s):
    
    k = np.arange(-N/2,N/2)
    Ff = (np.random.randn(N) + np.random.randn(N)*np.complex(0,1))/((k**2 + 1)**(s/2))
    Ff = fft.fftshift(Ff)
    f = fft.ifft(Ff)*N
    
    return f

class Animate():
    '''Used to define an animate object used to animate the solution. These
    objects store the current u0 value, after nonlinear_CGL_solution is called, in a way 
    sort of replicating a static function variable. If you want to c
    compute the solution at time t_2 and have the value at t_1 you can go 
    from t_1 to t_2 instead of 0 to t_2 which would be inefficent.
    '''
    def __init__(self, line1, line2, dt, Nt, u_0, a_1, a_2, b_1, b_2, c_1, c_2):
        self.line1 = line1
        self.line2 = line2
        self.dt = dt
        self.Nt = Nt
        self.u = u_0
        self.a_1 = a_1
        self.a_2 = a_2
        self.b_1 = b_1
        self.b_2 = b_2
        self.c_1 = c_1
        self.c_2 = c_2
        self.x = np.linspace(0,1,len(u_0)+1)
        self.x = np.delete(self.x,-1)
        
    def __call__(self,i):
        self.u = nonlinear_CGL_solution(self.dt, self.Nt, self.u, self.a_1, self.a_2, 
                                        self.b_1, self.b_2, self.c_1, self.c_2) 
        self.line1.set_data(self.x, np.real(self.u))
        self.line2.set_data(self.x, np.imag(self.u))
        
        return self.line1, self.line2
    

if __name__ == "__main__": main()
    
    
    

