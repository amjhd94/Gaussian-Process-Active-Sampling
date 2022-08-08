import numpy as np
from gpsearch.core import (BlackBox, UniformInputs, OptimalDesign, custom_KDE)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#%% Objective functions

def map_def_hopf(X, u_init=[.2, -.1]):
    x = 0*(X<=0) + np.sqrt(np.abs(X))*(X>0) + .000001*X
    return x

def map_def_transcritical_art(X):
    tc_bd = -4*X**2 + (.5*(X<=0) + 20*(X>0))*X
    tc_bd = 0.04878048780487805*tc_bd
    return tc_bd

#%% Defining the problem and intial dataset

my_map = BlackBox(map_def_transcritical_art, args=())

np.random.seed(12512)
n_init = 2
n_iter = 30

domain = [[-1, 1]]


inputs = UniformInputs(domain)


X_plot = inputs.draw_samples(int(1e4), "grd")
Y_plot = my_map.evaluate(X_plot)

ww = inputs.pdf(X_plot)

pdf_orig = custom_KDE(Y_plot, weights=ww, bw=.05)
pdf_orig_x, pdf_orig_y = pdf_orig.evaluate()

X = inputs.draw_samples(n_init, "uni")
Y = my_map.evaluate(X)

plt.figure()
plt.subplot(1,2,1)
plt.plot(X_plot, Y_plot)
plt.scatter(X, Y, s=200, c='y')
plt.subplot(1,2,2)
plt.plot(pdf_orig_x, pdf_orig_y)

#%% Active sampling

acq_list = ['my_acq1__0_0_1000', 'IVR_LW', 'us_lw']

acq_func_idx = 2

acquisition = acq_list[acq_func_idx]

o = OptimalDesign(X, Y, my_map, inputs, 
                  fix_noise=True, 
                  noise_var=0.0, 
                  normalize_Y=True)

m_list = o.optimize(n_iter,
                    acquisition=acquisition,
                    num_restarts=10,
                    parallel_restarts=True,
                    prefix=acquisition+"obj_fcn_")

#%%

log_pdf_error = []
R2_map = []
for i in range(0, n_iter+1, 1):
    print(i)
    model_i = m_list[i]
    
    mean, _ = model_i.predict(X_plot)
    
    pdf_model_data = custom_KDE(mean, weights=inputs.pdf(X_plot), bw=0.05)
    
    pdf_model_data_x, pdf_model_data_y = pdf_model_data.evaluate()
    
    x_min = min( pdf_model_data.data.min(), pdf_orig.data.min() )
    x_max = max( pdf_model_data.data.max(), pdf_orig.data.max() )
    rang = x_max-x_min
    x_eva = np.linspace(x_min - 0.01*rang,
                        x_max + 0.01*rang, 1024)
    
    yb, yt = pdf_model_data.evaluate(x_eva), pdf_orig.evaluate(x_eva)
    log_yb, log_yt = np.log(yb), np.log(yt)
    
    np.clip(log_yb, -3, None, out=log_yb)
    np.clip(log_yt, -3, None, out=log_yt)
    
    log_diff = np.abs(log_yb-log_yt)
    noInf = np.isfinite(log_diff)
    
    log_pdf_error.append(np.trapz(log_diff[noInf], x_eva[noInf]))  
    
    
    corr_matrix_map = np.corrcoef(Y_plot.flatten(), mean.flatten())
    corr_map = corr_matrix_map[0,1]
    R_sq_map = corr_map**2
    R2_map.append(R_sq_map)
    
    if i%10 == 0:
        plt.figure(dpi=200)
        plt.subplot(1,2,1)
        plt.plot(np.sort(X_plot), Y_plot, 'k', lw=4, label='True function')
        plt.plot(X_plot, mean, label='Model')
        plt.scatter(m_list[i].X[0:n_init], m_list[i].Y[0:n_init], c='y', label='Initial inputs')
        plt.scatter(m_list[i].X[n_init:], m_list[i].Y[n_init:], c='r', label='Acquired samples')
        plt.title('iter '+str(i))
        plt.legend(loc=2)
        
        plt.subplot(1,2,2)
        plt.semilogy(pdf_orig_x, pdf_orig_y, 'k', lw=4, label='True function')
        plt.semilogy(pdf_model_data_x, pdf_model_data_y, 'r', label='Model')
        plt.ylim(bottom=1e-2)
        plt.legend(loc=2)
    
    
plt.figure()
plt.subplot(1,2,1)
plt.semilogy(np.array(log_pdf_error))
plt.xlabel('Iterations')
plt.ylabel('log pdf error')
plt.subplot(1,2,2)
plt.plot(np.array(R2_map))
plt.xlabel('Iterations')
plt.ylabel('functions $R^2$')
