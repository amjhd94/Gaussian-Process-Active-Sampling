import numpy as np
import time
import GPy
from gpsearch.core.MyModel import MyModelWithHessian
from .utils import set_worker_env
from .minimizers import funmin
from .kernels import *
from .acquisitions.check_acquisition import check_acquisition
from gpsearch.core.likelihood import Likelihood
import matplotlib.pyplot as plt

class OptimalDesign_FixedDS(object):
    """A class for Bayesian sequential-search algorithm on fixed datasets.

    Parameters
    ---------
    X : array
        Array of input points.
    Y : array
        Array of observations.
    my_map: instance of `BlackBox`
        The black-box objective function.
    inputs : instance of `Inputs`
        The input space.
    fix_noise : boolean, optional
        Whether or not to fix the noise variance in the GP model.
    noise_var : float, optional
        Variance for additive Gaussian noise. Default is None, in 
        which case the noise variance from BlackBox is used. If fix_noise 
        is False, noise_var is merely used to initialize the GP model.
    normalize_Y : boolean, optional
        Whether or not to normalize the output in the GP model. 

    Attributes
    ----------
    X, Y, my_map, inputs : see Parameters
    input_dim : int
        Dimensionality of the input space
    model : instance of `GPRegression`
        Current GPy model.

    """

    def __init__(self, X_init, Y_init, inputs, fix_noise=False, 
                 noise_var=None, normalize_Y=True):

        
        self.inputs = inputs
        self.input_dim = inputs.input_dim

        self.X = np.atleast_2d(X_init)
        self.Y = np.atleast_2d(Y_init)

        if noise_var is None:
            noise_var = 0

        # Currently only the RBF kernel is supported.
        ker = RBF(input_dim=self.input_dim, ARD=True)

        # self.model = GPy.models.GPRegression(X=self.X, 
        #                                      Y=self.Y, 
        #                                      kernel=ker, 
        #                                      normalizer=normalize_Y,
        #                                      noise_var=noise_var)
        
        self.model = MyModelWithHessian(X=self.X, 
                                        Y=self.Y, 
                                        kernel=ker, 
                                        normalizer=normalize_Y,
                                        noise_var=noise_var)
        
        # pts = inputs.draw_samples(n_samples=int(1e6), sample_method="grd")

        # mu, sig = self.model.predict(pts)
        
        # plt.figure(dpi=200)
        # plt.scatter(self.X, self.Y)
        # plt.plot(pts, mu)
        # plt.plot(pts, sig/np.max(sig))
        
        if fix_noise:
            self.model.Gaussian_noise.variance.fix(noise_var)

    def optimize(self, n_iter, X_DS, Y_DS, acquisition, prefix=None, 
                 callback=True, save_iter=True, kwargs_GPy=None):
        """Runs the Bayesian sequential-search algorithm.

        Parameters
        ----------
        n_iter : int
            Number of iterations (i.e., black-box queries) to perform.
        acquisition : str or instance of `Acquisition`
            Acquisition function for determining the next best point.
            If a string, must be one of 
                - "PI": Probability of Improvement
                - "EI": Expected Improvement
                - "US": Uncertainty Sampling
                - "US_BO": US repurposed for Bayesian Optimization (BO)
                - "US_LW": Likelihood-Weighted US
                - "US_LWBO": US-LW repurposed for BO
                - "US_LWraw" : US-LW with no GMM approximation
                - "US_LWBOraw": US-LWBO with no GMM approximation
                - "LCB" : Lower Confidence Bound
                - "LCB_LW" : Likelihood-Weighted LCB
                - "LCB_LWraw" : LCB-LW with no GMM approximation
                - "IVR" : Integrated Variance Reduction
                - "IVR_IW" : Input-Weighted IVR
                - "IVR_LW" : Likelihood-Weighted IVR
                - "IVR_BO" : IVR repurposed for BO
                - "IVR_LWBO": IVR-LW repurposed for BO
                - "IVR_HO" : Higher-order IVR
        opt_method : {"L-BFGS-B", "SLSQP", "TNC"}, optional 
            Type of solver. 
        num_restarts : int, optional
            Number of restarts for the optimizer (see funmin).
        parallel_restarts : boolean, optional
            Whether or not to perform optimization in parallel.
        n_jobs : int, optional
            Number of workers used by joblib for parallel computation.
        callback : boolean, optional
            Whether or not to display log at each iteration.
        save_iter : boolean, optional
            Whether or not to save the GP model at each iteration.
        prefix : string, optional
            Prefix for file naming
        kwargs_GPy : dict, optional
            Dictionary of arguments to be passed to the GP model.
        kwargs_op : dict, optional
            Dictionary of arguments to be passed to the scipy optimizer.
      
        Returns
        -------
        m_list : list
            A list of trained GP models, one for each iteration of 
            the algorithm.

        """
        
        if prefix==None:
            prefix = 'FixedDS_Model'
            
        if kwargs_GPy is None:
            kwargs_GPy = dict(num_restarts=10, optimizer="bfgs", 
                              max_iters=1000, verbose=False) 
            # kwargs_GPy = dict(verbose=False) 
        pts_0 = self.inputs.draw_samples(n_samples=int(1e4), sample_method="grd")
        m_list = []
        mean_sig = []
        acq = check_acquisition(acquisition, self.model, self.inputs)
        
        # pts2 = self.inputs.draw_samples(n_samples=int(200), sample_method="grd")

        for ii in range(n_iter+1):

            filename = (prefix+"%.4d")%(ii)

            if ii == 0:
                self.model.optimize_restarts(**kwargs_GPy)
                m_list.append(self.model.copy())
                if save_iter:
                    self.model.save_model(filename)
                continue

            tic = time.time()

            acq.model = self.model.copy()
            acq.update_parameters()
            # if ii == n_iter:
            #     plt.figure()
            #     plt.plot(X_DS, acq.model.predict(X_DS)[1].flatten())
            _, var_DS = self.model.predict(X_DS)
            try:
                if "my_acq1__0" in acquisition.lower():
                    likelihood1 = acq.a1.likelihood.evaluate(X_DS)
                    likelihood2 = acq.a2.likelihood.evaluate(X_DS)
                    acq_evaluate = (likelihood2*var_DS)**2
                elif "my_acq1_0" in acquisition.lower():
                    likelihood1 = acq.a1.likelihood.evaluate(X_DS)
                    likelihood2 = acq.a2.likelihood.evaluate(X_DS)
                    acq_evaluate = (likelihood2*var_DS)**2
                elif "my_acq1__1" in acquisition.lower():
                    likelihood1 = acq.a1.likelihood.evaluate(X_DS)
                    likelihood2 = acq.a2.likelihood.evaluate(X_DS)
                    acq_evaluate = likelihood1*var_DS + (likelihood2*var_DS)**2
                elif "my_acq1_1" in acquisition.lower():
                    likelihood1 = acq.a1.likelihood.evaluate(X_DS)
                    likelihood2 = acq.a2.likelihood.evaluate(X_DS)
                    acq_evaluate = likelihood1*var_DS + (likelihood2*var_DS)**2
                elif "lw" in acquisition.lower():
                    likelihood1 = acq.likelihood.evaluate(X_DS)
                    acq_evaluate = likelihood1*var_DS
            except:
                likelihood = np.ones(X_DS.shape)
                acq_evaluate = likelihood*var_DS
                
            
            
            
            # xopt_idx = np.argmin(acq.evaluate(X_DS))
            xopt_idx = np.argmax(acq_evaluate)
            

            xopt = np.atleast_2d(X_DS[xopt_idx])
            yopt = Y_DS[xopt_idx]

            self.X = np.vstack((self.X, xopt))
            self.Y = np.vstack((self.Y, yopt))
            self.model.set_XY(self.X, self.Y)
            self.model.optimize_restarts(**kwargs_GPy)
            m_list.append(self.model.copy())
            _, var = self.model.predict(pts_0)
            sig = np.sqrt(var)
            sig_avg = np.sum(sig)/len(sig)
            mean_sig.append(sig_avg)
            
            # if ii == n_iter:
            #     plt.figure()
            #     acq = check_acquisition(acquisition, self.model, self.inputs)  
            #     acq_func_norm = -acq.evaluate(pts2)/np.max(-acq.evaluate(pts2))
            #     mu_ii, var_ii = self.model.predict(pts2)
            #     likelihood1 = Likelihood(self.model, self.inputs, weight_type="importance")
            #     IVR_LW_likelihood = likelihood1.evaluate(pts2)
            #     IVR_LW_likelihood_norm = IVR_LW_likelihood/np.max(IVR_LW_likelihood)
            #     plt.plot(pts2, mu_ii, label='mu_'+str(ii))
            #     plt.plot(pts2, var_ii/np.max(var_ii), label='var_'+str(ii))
            #     plt.plot(pts2, acq_func_norm + 2, label='acq_'+str(ii))
            #     plt.plot(pts2, IVR_LW_likelihood_norm + 2, label='LW_'+str(ii))
            #     plt.scatter(self.model.X, self.model.Y, label='smp_'+str(ii))
            #     plt.legend()
            

            if callback:
                self._callback(ii, time.time()-tic)

            if save_iter:
                self.model.save_model(filename)

        return m_list, mean_sig

    @staticmethod
    def _callback(ii, time):
         m, s = divmod(time, 60)
         print("Iteration {:3d} \t Optimization completed in {:02d}:{:02d}"
               .format(ii, int(m), int(s)))