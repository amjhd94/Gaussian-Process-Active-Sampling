import numpy as np
import time
from gpsearch.core import (BlackBox, GaussianInputs, UniformInputs, OptimalDesign, 
                       OptimalDesign_FixedDS, custom_KDE, model_pdf)
from gpsearch.core.acquisitions import *
from gpsearch.core.acquisitions.check_acquisition import check_acquisition
from gpsearch.core.likelihood import Likelihood
import GPy
from gpsearch.core.MyModel import MyModelWithHessian
from gpsearch.core.kernels import *
import matplotlib.pyplot as plt

np.random.seed(110)

def Hopf_bif_func(mu):
    x = 0*(mu<=0) + np.sqrt(np.abs(mu))*(mu>0) + .000001*mu
    return x

def map_def_transcritical_art(X):
    tc_bd = -4*X**2 + (.5*(X<=0) + 20*(X>0))*X
    tc_bd = 0.04878048780487805*tc_bd
    return tc_bd

def test(X):
    x = 10*np.abs(X)
    return x

def test2(X):
    x = X**3 - 2*X
    return x

mu = np.linspace(-2,2,int(200))
x0 = Hopf_bif_func(mu)
sig_n = 0.00
x0 = x0 + np.random.normal(0,sig_n,len(mu)).reshape(x0.shape)

Y_DS0 = np.atleast_2d(x0).T
X_DS0 = np.atleast_2d(mu).T

Y_DS = np.atleast_2d(x0).T
X_DS = np.atleast_2d(mu).T


domain = np.zeros((X_DS.shape[1], 2))
domain[:,0] = np.min(X_DS)
domain[:,1] = np.max(X_DS)

X_init = X_DS
Y_init = Y_DS

ker = RBF(input_dim=X_DS.shape[1], ARD=True)

model_test = MyModelWithHessian(X=X_init, 
                            Y=Y_init, 
                            kernel=ker,
                            normalizer=True,
                            noise_var=0)

inputs = UniformInputs(domain)

ww = inputs.pdf(X_DS0)
bw = .025
pdf_DS = custom_KDE(Y_DS0, bw=bw)
Y_DS_output, Y_DS_pdf = pdf_DS.evaluate()

# plt.figure()
# plt.semilogy(Y_DS_output, Y_DS_pdf)
# plt.xlabel('$r = \sqrt{x^2 + y^2}$')
# plt.ylabel('$p_r(r)$')
# # plt.ylim([2e-1, 10])

kwargs_GPy = dict(num_restarts=10, optimizer="bfgs", 
                              max_iters=1000, verbose=False) 
model_test.optimize_restarts(**kwargs_GPy)
#%
inputs = UniformInputs(domain)
pts = inputs.draw_samples(n_samples=int(1e6), sample_method="grd")

mu, var = model_test.predict(pts)
sig = np.sqrt(var)

# plt.figure()
# plt.plot(pts, mu)
# plt.fill_between(pts.flatten(), (mu-sig).flatten(), (mu+sig).flatten(), alpha=0.2)

acq_list = ['my_acq1__0_0_1000', 'IVR_LW']

i = 1

acquisition = acq_list[i]

pts2 = inputs.draw_samples(n_samples=int(1e3), sample_method="grd")

acq = check_acquisition(acquisition, model_test, inputs)  
acq_func_norm = -acq.evaluate(pts2)/np.max(-acq.evaluate(pts2))


if i == 0:
    likelihood2 = acq.a2.likelihood.evaluate(pts2)
    likelihood = likelihood2
    likelihood_norm = likelihood/np.max(likelihood)
elif i == 1:
    likelihood1 = acq.likelihood.evaluate(pts2)
    likelihood = likelihood1
    likelihood_norm = likelihood/np.max(likelihood)

_, var2 = model_test.predict(pts2)

# plt.figure()
# plt.plot(pts, mu)
# plt.plot(pts2, acq_func_norm, label='acq1')
# plt.plot(pts2, likelihood_norm, label='lik')
# plt.plot(pts2, var2/np.max(var2), label='var')
# plt.plot(pts2, likelihood*var2/np.max(likelihood*var2), label='acq2')
# plt.legend()

X_important = []
Y_important = []

n_iter = 10
n_importnat = 5

if i == 0:
    likelihood2 = acq.a2.likelihood.evaluate(X_DS)
    likelihood = likelihood2
    likelihood_norm = likelihood/np.max(likelihood)
elif i == 1:
    likelihood1 = acq.likelihood.evaluate(X_DS)
    likelihood = likelihood1
    likelihood_norm = likelihood/np.max(likelihood)

likelihood_norm_conv = likelihood_norm

sorting_indices = np.argsort(likelihood_norm.flatten())

X_important.append(X_DS[sorting_indices[-(n_importnat-1):]])
Y_important.append(Y_DS[sorting_indices[-(n_importnat-1):]])



X_DS = np.delete(X_DS, sorting_indices[-(n_importnat-1):], 0)
Y_DS = np.delete(Y_DS, sorting_indices[-(n_importnat-1):], 0)

X_init = np.array(X_important)[0,:,:]
Y_init = np.array(Y_important)[0,:,:]

m_list = []

# plt.figure()
# plt.scatter(X_DS, Y_DS)
#%
Iter = 0
logPDF_diff = []
logPDF_R2 = []
map_R2 = []
sample_NO = []
pts3 = inputs.draw_samples(n_samples=int(1e3), sample_method="grd")
# while len(Y_DS)>n_importnat:
for iteration in range(n_iter):
    
    model = MyModelWithHessian(X=X_init, 
                            Y=Y_init, 
                            kernel=ker,
                            normalizer=True,
                            noise_var=0)    
    model.optimize_restarts(**kwargs_GPy)
    
    mu0, _ = model_test.predict(X_DS0)    
    
    mu_pdf, _ = model.predict(pts3)
    pdf_mu = custom_KDE(mu_pdf, bw=bw)
    mu_output, mu_pdf = pdf_mu.evaluate()
    x_min = min( pdf_mu.data.min(), pdf_DS.data.min() )
    x_max = max( pdf_mu.data.max(), pdf_DS.data.max() )
    rang = x_max-x_min
    x_eva = np.linspace(x_min - 0.01*rang,
                        x_max + 0.01*rang, 1024)
    
    yb, yt = pdf_mu.evaluate(x_eva), pdf_DS.evaluate(x_eva)
    log_yb, log_yt = np.log(yb), np.log(yt)
    
    np.clip(log_yb, -14, None, out=log_yb)
    np.clip(log_yt, -14, None, out=log_yt)
    
    log_diff = np.abs(log_yb-log_yt)
    noInf = np.isfinite(log_diff)
    
    logPDF_diff.append(np.trapz(log_diff[noInf], x_eva[noInf]))
    
    corr_matrix_map = np.corrcoef(Y_DS0.flatten(), mu0.flatten())
    corr_map = corr_matrix_map[0,1]
    R_sq_map = corr_map**2
    map_R2.append(R_sq_map)
    
    corr_matrix_map = np.corrcoef(log_yb.flatten(), log_yt.flatten())
    corr_map = corr_matrix_map[0,1]
    R_sq_logpdf = corr_map**2
    logPDF_R2.append(R_sq_logpdf)
    
    smpl_NO = len(X_init)
    sample_NO.append(smpl_NO)
    
    
    if i == 0:
        likelihood2 = acq.a2.likelihood.evaluate(X_DS0)
        likelihood = likelihood2
        likelihood_norm = likelihood/np.max(likelihood)
    elif i == 1:
        likelihood1 = acq.likelihood.evaluate(X_DS0)
        likelihood = likelihood1
        likelihood_norm = likelihood/np.max(likelihood)
    
    mu_DS, var = model.predict(X_DS0)
    # plt.figure()
    # plt.scatter(np.log10(var), np.log10(likelihood))
    # plt.xlim([-50, 2])
    # plt.ylim([-50, 2])
    # plt.savefig('new_DQA_fig\Fig_of_Model00'+str(N_iter)+'_'+acquisition+'.png')
    # plt.close()
    # if Iter%4 == 0:
    #     plt.figure(dpi=200)
    #     plt.subplot(2,2,1)
    #     plt.scatter(X_DS0, Y_DS0, s=50, c='k', label='Original Data')
    #     plt.scatter(np.array(X_important)[0:(Iter+1), :, 0].flatten(), np.array(Y_important)[0:(Iter+1), :, 0].flatten(), s=25, c='r', label='Selected Data')
    #     plt.legend()
    #     plt.subplot(2,2,3)
    #     plt.hist(np.array(X_important)[0:(Iter+1), :, 0].flatten(), bins=list(np.linspace(-2, 2, 25)), label='Selected samples dist.')
    #     plt.plot(X_DS0, likelihood_norm_conv*4, label='Likelihood function (gmm approx.)')
    #     plt.legend()
    #     plt.subplot(2,2,2)
    #     plt.semilogy(Y_DS_output, Y_DS_pdf, 'k', label='True pdf')
    #     plt.semilogy(mu_output, mu_pdf, 'r', label='Model pdf')
    #     plt.xlim([-0.85, 1.2])
    #     plt.ylim([5e-2, 10])
    #     plt.legend()
    #     plt.subplot(2,2,4)
    #     plt.plot(X_DS0, Y_DS0, label='obj. fcn.')
    #     plt.plot(X_DS0, mu_DS, label='model')
    #     plt.legend()
    #     plt.savefig('new_DQA_fig\Data_of_Model00'+str(Iter)+'_'+acquisition+'.png')
    
    # plt.figure()
    # plt.scatter(X_DS0, Y_DS0)
    # plt.scatter(m_list[N_iter].X, m_list[N_iter].Y)
    
    acq = check_acquisition(acquisition, model, inputs) 
    
    if i == 0:
        likelihood2 = acq.a2.likelihood.evaluate(X_DS)
        likelihood = likelihood2
        likelihood_norm = likelihood/np.max(likelihood)
    elif i == 1:
        likelihood1 = acq.likelihood.evaluate(X_DS)
        likelihood = likelihood1
        likelihood_norm = likelihood/np.max(likelihood)

    _, sig_DS = model.predict(X_DS)
    acq_val = likelihood*sig_DS

    # sorting_indices = np.argsort(likelihood_norm.flatten())
    sorting_indices = np.argsort(acq_val.flatten())
    
    X_important.append(X_DS[sorting_indices[-(n_importnat-1):]])
    Y_important.append(Y_DS[sorting_indices[-(n_importnat-1):]])
       
    X_DS = np.delete(X_DS, sorting_indices[-(n_importnat-1):], 0)
    Y_DS = np.delete(Y_DS, sorting_indices[-(n_importnat-1):], 0)
    
    X_init = np.atleast_2d(np.array(X_important).flatten()).T
    Y_init = np.atleast_2d(np.array(Y_important).flatten()).T
    
    # plt.plot(pts2, model.predict(pts2)[0], label=('iter: '+str(Iter)))
    # plt.legend()
    print(Iter)
    del model
    Iter += 1

# np.save('new_DQA_fig\log_diff_'+acquisition, np.array(logPDF_diff))
# np.save('new_DQA_fig\logpdf_R2_'+acquisition, np.array(logPDF_R2))
# np.save('new_DQA_fig\map_R2_'+acquisition, np.array(map_R2))
# np.save('new_DQA_fig\sample_NO_'+acquisition, np.array(sample_NO))
plt.figure(dpi=200)
plt.subplot(1,3,1)
plt.plot(np.array(sample_NO)/200*100, np.array(logPDF_diff), '-o')
plt.subplot(1,3,2)
plt.plot(np.array(sample_NO)/200*100, np.array(logPDF_R2), '-o')
plt.subplot(1,3,3)
plt.plot(np.array(sample_NO)/200*100, np.array(map_R2), '-o')
#%%
# Ivrlw_logpdfdiff = np.load("C:\\MIT\\Codes\\Gaussian Process project\\Mine\\gpsearch-master\\gpsearch\\examples\\test_bif\\new_DQA_fig\\Hopf_old_likelihood\\log_diff_IVR_LW.npy")
# Ivrlw_logpdfR2 = np.load("C:\\MIT\\Codes\\Gaussian Process project\\Mine\\gpsearch-master\\gpsearch\\examples\\test_bif\\new_DQA_fig\\Hopf_old_likelihood\\logpdf_R2_IVR_LW.npy")
# Ivrlw_mapR2 = np.load("C:\\MIT\\Codes\\Gaussian Process project\\Mine\\gpsearch-master\\gpsearch\\examples\\test_bif\\new_DQA_fig\\Hopf_old_likelihood\\map_R2_IVR_LW.npy")
# Ivrlw_smpno = np.load("C:\\MIT\\Codes\\Gaussian Process project\\Mine\\gpsearch-master\\gpsearch\\examples\\test_bif\\new_DQA_fig\\Hopf_old_likelihood\\sample_NO_IVR_LW.npy")

# Ivrlgw_logpdfdiff = np.load("C:\\MIT\\Codes\\Gaussian Process project\\Mine\\gpsearch-master\\gpsearch\\examples\\test_bif\\new_DQA_fig\\Hopf_old_likelihood\\log_diff_my_acq1__0_0_1000.npy")
# Ivrlgw_logpdfR2 = np.load("C:\\MIT\\Codes\\Gaussian Process project\\Mine\\gpsearch-master\\gpsearch\\examples\\test_bif\\new_DQA_fig\\Hopf_old_likelihood\\logpdf_R2_my_acq1__0_0_1000.npy")
# Ivrlgw_mapR2 = np.load("C:\\MIT\\Codes\\Gaussian Process project\\Mine\\gpsearch-master\\gpsearch\\examples\\test_bif\\new_DQA_fig\\Hopf_old_likelihood\\map_R2_my_acq1__0_0_1000.npy")
# Ivrlgw_smpno = np.load("C:\\MIT\\Codes\\Gaussian Process project\\Mine\\gpsearch-master\\gpsearch\\examples\\test_bif\\new_DQA_fig\\Hopf_old_likelihood\\sample_NO_my_acq1__0_0_1000.npy")


# plt.figure(dpi=200)
# plt.subplot(1,3,1)
# plt.semilogy(Ivrlgw_smpno*100/200, Ivrlgw_logpdfdiff, '-or', label='IVR-LGW')
# plt.semilogy(Ivrlw_smpno*100/200, Ivrlw_logpdfdiff, '-ok', label='IVR-LW')
# plt.xlim([0, 100])
# plt.ylabel('$Error$')
# plt.xlabel('$Data$ $percentage$')

# plt.subplot(1,3,2)
# plt.plot(Ivrlgw_smpno*100/200, Ivrlgw_logpdfR2, '-or', label='IVR-LGW')
# plt.plot(Ivrlw_smpno*100/200, Ivrlw_logpdfR2, '-ok', label='IVR-LW')
# plt.ylim([-.05, 1.05])
# plt.xlim([0, 100])
# plt.ylabel('$R^2$ $-$ $log$ $pdfs$')
# plt.xlabel('$Data$ $percentage$')

# plt.subplot(1,3,3)
# plt.plot(Ivrlgw_smpno*100/200, Ivrlgw_mapR2, '-or', label='IVR-LGW')
# plt.plot(Ivrlw_smpno*100/200, Ivrlw_mapR2, '-ok', label='IVR-LW')
# plt.ylim([-.05, 1.05])
# plt.xlim([0, 100])
# plt.ylabel('$R^2$ $-$ $maps$')
# plt.xlabel('$Data$ $percentage$')
# plt.legend()