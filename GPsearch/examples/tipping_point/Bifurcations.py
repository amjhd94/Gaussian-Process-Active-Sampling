import numpy as np
# import matplotlib.pyplot as plt


class Bifurcations:

    def __init__(self, X, u_init, tf=1000, nsteps=10000):
        self.mu = X
        self.tf = tf
        self.nsteps = nsteps
        self.u_init = u_init

    def transcritical(self, u, t):
        f = self.mu*u - u**2
        return f
    
    def pitchfork(self, u, t):
        f = self.mu*u - u**3
        return f
    
    def hopf(self, u, t):
        u0, u1 = u
        f0 = self.mu*u0 - u1 - u0*(u0**2 + u1**2)
        f1 = u0 + self.mu*u1 - u1*(u0**2 + u1**2)
        f = [f0, f1]
        return f
    
    def solve(self, DS):
        time = np.linspace(0, self.tf, self.nsteps+1)
        solver = ODESolver(DS)
        solver.set_ics(self.u_init)
        u, t = solver.solve(time)
        return u, t


class ODESolver:

    def __init__(self, f):
        self.f = lambda u, t: np.asarray(f(u, t), float)

    def set_ics(self, U0):
        U0 = np.asarray(U0)
        self.neq = U0.size
        self.U0 = U0

    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        
        K1 = dt*f(u[k], t[k]).flatten()
        K2 = dt*f(u[k] + 0.5*K1, t[k] + 0.5*dt).flatten()
        K3 = dt*f(u[k] + 0.5*K2, t[k] + 0.5*dt).flatten()
        K4 = dt*f(u[k] + K3, t[k] + dt).flatten()
        u_new = u[k] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        return u_new

    def solve(self, time):
        self.t = np.asarray(time)
        n = self.t.size
        self.u = np.zeros((n,self.neq))
        self.u[0] = self.U0
        for k in range(n-1):
            self.k = k
            self.u[k+1] = self.advance()
        return self.u[:k+2], self.t[:k+2]
