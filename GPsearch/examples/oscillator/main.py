import numpy as np
from gpsearch import (BlackBox, GaussianInputs, OptimalDesign, custom_KDE)
from oscillator import Oscillator, Noise
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
sns.set()

#%% Objective function

def map_def(theta, oscil):
    u, t = oscil.solve(theta)
    mean_disp = np.mean(u[:,0])
    return mean_disp

#%% Defining the problem and intial dataset

ndim = 2
np.random.seed(3)

tf = 25
nsteps = 1000
u_init = [0, 0]
noise = Noise([0, tf])
oscil = Oscillator(noise, tf, nsteps, u_init)
my_map = BlackBox(map_def, args=(oscil,))

n_init = 4
n_iter = 20

mean, cov = np.zeros(ndim), np.ones(ndim)

domain = [ [-6, 6] ] * ndim

inputs = GaussianInputs(domain, mean, cov)

X = inputs.draw_samples(n_init, "lhs")
Y = my_map.evaluate(X)

#%% Active sampling

o = OptimalDesign(X, Y, my_map, inputs, 
                  fix_noise=True, 
                  noise_var=0.0, 
                  normalize_Y=True)
m_list = o.optimize(n_iter,
                    acquisition="US_LW",
                    num_restarts=10,
                    parallel_restarts=True)

#%% Reference data to compute true pdf

filename = "map_samples{:d}D.txt".format(ndim)
try:
    smpl = np.genfromtxt(filename)
    pts = smpl[:,0:-1]
    yy = smpl[:,-1]
except:
    pts = inputs.draw_samples(n_samples=100, sample_method="grd")
    yy = my_map.evaluate(pts, parallel=False, include_noise=False)
    np.savetxt(filename, np.column_stack((pts,yy)))

pdf = custom_KDE(yy, weights=inputs.pdf(pts), bw=.05)
pdf_x, pdf_y = pdf.evaluate()

#%%

n_samples = int(1e2)
x_test = pts
y_test = yy
log_pdf_error = []
R2_map = []
for i in range(0, n_iter+1, 1):
    print(i)
    ens_model_i = m_list[i]
    
    mean, _ = ens_model_i.predict(x_test)
    
    pdf_model_data = custom_KDE(mean, weights=inputs.pdf(pts), bw=0.05)
    
    x_min = min( pdf_model_data.data.min(), pdf.data.min() )
    x_max = max( pdf_model_data.data.max(), pdf.data.max() )
    rang = x_max-x_min
    x_eva = np.linspace(x_min - 0.01*rang,
                        x_max + 0.01*rang, 1024)
    
    yb, yt = pdf_model_data.evaluate(x_eva), pdf.evaluate(x_eva)
    log_yb, log_yt = np.log(yb), np.log(yt)
    
    np.clip(log_yb, -14, None, out=log_yb)
    np.clip(log_yt, -14, None, out=log_yt)
    
    log_diff = np.abs(log_yb-log_yt)
    noInf = np.isfinite(log_diff)
    
    log_pdf_error.append(np.trapz(log_diff[noInf], x_eva[noInf]))
    
    corr_matrix_map = np.corrcoef(y_test.flatten(), mean.flatten())
    corr_map = corr_matrix_map[0,1]
    R_sq_map = corr_map**2
    R2_map.append(R_sq_map)
    
    if i%5 == 0:
        plt.figure()
        plt.subplot(1,2,1)
        triang = tri.Triangulation(pts[:,0].flatten(), pts[:,1].flatten())
        interpolator = tri.LinearTriInterpolator(triang, yy.flatten())
        plt.tricontourf(pts[:,0].flatten(), pts[:,1].flatten(), yy.flatten(), levels=30)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.colorbar()
        plt.plot(m_list[i].X[0:n_init,0], m_list[i].X[0:n_init,1], 'xw', label='Initial inputs')
        plt.plot(m_list[i].X[n_init:,0], m_list[i].X[n_init:,1], '.w', label='Acquired samples')
        plt.legend
        plt.subplot(1,2,2)
        plt.semilogy(x_eva, yt, label='True pdf')
        plt.semilogy(x_eva, yb, label='Model pdf')
        plt.legend()
        
    
    
plt.figure()
plt.plot(np.array(log_pdf_error))
plt.xlabel('Iterations')
plt.ylabel('log pdf error')


