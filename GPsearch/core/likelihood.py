import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.mixture import GaussianMixture as GMM
from .utils import fix_dim_gmm, custom_KDE
import numpy as np
import matplotlib.pyplot as plt
import math


class Likelihood(object):
    """A class for computation of the likelihood ratio.

    Parameters
    ----------
    model : instance of GPRegression
        A GPy model 
    inputs : instance of Inputs
        The input space.
    weight_type : str, optional
        Type of likelihood weight. Must be one of
            - "nominal" : uses w(x) = p(x)
            - "importance" : uses w(x) = p(x)/p_y(mu(x))
            - "importance_H.O." # Alireza Mojahed
    fit_gmm : boolean, optional
        Whether or not to use a GMM approximation for the likelihood
        ratio.  
    kwargs_gmm : dict, optional
        A dictionary of keyword arguments for scikit's GMM routine.
        Use this to specify the number of Gaussian mixtures and the
        type of covariance matrix.

    Attributes
    ----------
    model, inputs, weight_type, fit_gmm, kwargs_gmm : see Parameters
    fy_interp : scipy 1-D interpolant
        An interpolant for the output pdf p_y(mu)
    gmm : scikit Gaussian Mixture Model
        A GMM object approximating the likelihood ratio.

    """

    def __init__(self, model, inputs, weight_type="importance", 
                 fit_gmm=True, kwargs_gmm=None, trig_id=True, const_w=1e5, c_w2=1, c_w3=1, tol=1e-5):

        self.model = model
        self.inputs = inputs
        self.weight_type = self.check_weight_type(weight_type)
        self.fit_gmm = fit_gmm
        self.const_w = const_w
        self.trig_id = trig_id
        self.c_w2 = c_w2
        self.c_w3 = c_w3
        self.tol = tol

        if kwargs_gmm is None:
            # kwargs_gmm = dict(n_components=8, covariance_type="full")
            kwargs_gmm = dict(n_components=32, covariance_type="full")
        self.kwargs_gmm = kwargs_gmm

        self._prepare_likelihood()

    def update_model(self, model):
        self.model = model
        self._prepare_likelihood()
        return self

    def evaluate(self, x):
        """Evaluates the likelihood ratio at x.

        Parameters
        ----------
        x : array
            Query points. Should be of size (n_pts, n_dim)

        Returns
        -------
        w : array
            The likelihood ratio at x.

        """
        if self.fit_gmm:
            w = self._evaluate_gmm(x)
        else:
            w = self._evaluate_raw(x)
        return w

    def jacobian(self, x):
        """Evaluates the gradients of the likelihood ratio at x.

        Parameters
        ----------
        x : array
            Query points. Should be of size (n_pts, n_dim)

        Returns
        -------
        w_jac : array
            Gradients of the likelihood ratio at x.

        """
        if self.fit_gmm:
            w_jac = self._jacobian_gmm(x)
        else:
            w_jac = self._jacobian_raw(x)
        return w_jac

    def _evaluate_gmm(self, x):
        x = np.atleast_2d(x)
        w = np.exp(self.gmm.score_samples(x))
        return w[:,None]

    def _jacobian_gmm(self, x):
        x = np.atleast_2d(x)
        w_jac = np.zeros(x.shape)
        p = np.exp(self.gmm._estimate_weighted_log_prob(x))
        precisions = fix_dim_gmm(self.gmm, matrix_type="precisions")
        for ii in range(self.gmm.n_components):
            w_jac += p[:,ii,None] * np.dot(self.gmm.means_[ii]-x, \
                                           precisions[ii])
        return w_jac

    def _evaluate_raw(self, x):
        x = np.atleast_2d(x)
        fx = self.inputs.pdf(x)
        if self.weight_type == "nominal": 
            w = fx
        elif self.weight_type == "importance":
            mu = self.model.predict(x)[0].flatten()
            if self.model.normalizer:
                mu = self.model.normalizer.normalize(mu)
            fy = self.fy_interp(mu)
            w = fx/fy
        # """Addition by Alireza Mojahed"""
        elif self.weight_type == "importance_ho":
            mu = self.model.predict(x)[0].flatten()
            if self.model.normalizer:
                mu = self.model.normalizer.normalize(mu)
            fy = self.fy_interp(mu)
            w = fx/fy
        return w[:,None]

    def _jacobian_raw(self, x):
        x = np.atleast_2d(x)
        fx_jac = self.inputs.pdf_jac(x)

        if self.weight_type == "nominal":
            w_jac = fx_jac

        elif self.weight_type == "importance":
            mu = self.model.predict(x)[0].flatten()
            if self.model.normalizer:
                mu = self.model.normalizer.normalize(mu)
            mu_jac, _ = self.model.predictive_gradients(x)
            mu_jac = mu_jac[:,:,0]
            fx = self.inputs.pdf(x)
            fy = self.fy_interp(mu)
            fy_jac = self.fy_interp.derivative()(mu)
            tmp = fx * fy_jac / fy**2
            w_jac = fx_jac / fy[:,None] - tmp[:,None] * mu_jac
            
        # """Addition by Alireza Mojahed"""            
        elif self.weight_type == "importance_ho":
            mu = self.model.predict(x)[0].flatten()
            if self.model.normalizer:
                mu = self.model.normalizer.normalize(mu)
            mu_jac, _ = self.model.predictive_gradients(x)
            mu_jac = mu_jac[:,:,0]
            fx = self.inputs.pdf(x)
            fy = self.fy_interp(mu)
            fy_jac = self.fy_interp.derivative()(mu)
            tmp = fx * fy_jac / fy**2
            w_jac = fx_jac / fy[:,None] - tmp[:,None] * mu_jac
        

        return w_jac

    def _prepare_likelihood(self):
        """Prepare likelihood ratio for evaluation."""

        if self.inputs.input_dim <= 2:
            n_samples = int(1e5)
        else: 
            n_samples = int(1e6)

        pts = self.inputs.draw_samples(n_samples=n_samples, 
                                       sample_method="uni")
        fx = self.inputs.pdf(pts)

        if self.weight_type == "importance":
            mu = self.model.predict(pts)[0].flatten()
            if self.model.normalizer:
                mu = self.model.normalizer.normalize(mu)
            x, y = custom_KDE(mu, weights=fx).evaluate()
            self.fy_interp = InterpolatedUnivariateSpline(x, y, k=1)
            
        # """Addition by Alireza Mojahed"""
        if self.weight_type == "importance_ho":
            mu = self.model.predict(pts)[0].flatten()
            # mu_grad = self.model.predictive_gradients(pts)[1] # Check shape
            mu_grad = np.squeeze(self.model.predictive_gradients(pts)[0], axis=-1) # Check shape
            # mu_hess = self.model.predictive_hessian(pts) # Check shape
            mu_hess = self.model.predictive_hessian(pts)[:,:,:,0]
            # print('mu_grad shape is ',mu_grad.shape)
            # print('mu_hess shape is ',mu_hess.shape)
            
            if self.model.normalizer:
                mu = self.model.normalizer.normalize(mu)
            x, y = custom_KDE(mu, weights=fx).evaluate()
            self.fy_interp = InterpolatedUnivariateSpline(x, y, k=1)

        if self.fit_gmm:
            if self.weight_type == "nominal":
                w_raw = fx
                c0 = np.ones(fx.shape)
            elif self.weight_type == "importance":
                fy_jac = self.fy_interp.derivative()(mu)
                c0 = fx*np.abs(fy_jac)**2/(self.fy_interp(mu)**3)
                w_raw = fx/self.fy_interp(mu)
                
                # print(w_raw)
                
                # plt.figure()
                # plt.plot(w_raw[w_raw>.05],'r')
                
            # """Addition by Alireza Mojahed"""
            elif self.weight_type == "importance_ho":
                fy_jac = self.fy_interp.derivative()(mu)
                
                term_temp = np.array([np.sum(np.array([mu_grad[:,i]*mu_hess[:,i,j] for i in range(mu_grad.shape[1])]), axis=0) for j in range(mu_grad.shape[1])])
                term = np.sum(np.array([mu_grad[:,i]*term_temp.T[:,i] for i in range(mu_grad.shape[1])]),axis=0)
                term = term/(np.linalg.norm(mu_grad, axis=1)**4 + self.tol*self.c_w3)
                # term = np.empty(1)
                # for i in range(pts.shape[0]):
                #     d_mu = mu_grad[i,::]
                #     # H_mu = mu_hess[i,::].reshape((pts.shape[1],pts.shape[1]))
                #     H_mu = mu_hess[i,::]
                #     term = np.vstack((term,d_mu@H_mu@d_mu.T/(np.linalg.norm(d_mu)**4+self.tol)))
                    
                # term = term[1:].flatten()
                
                # print('term is computed!!')
                
                # print((self.c_w2*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4)).shape, (fx*np.abs(fy_jac)/(2*self.fy_interp(mu)**2)*term).shape, (fx*np.abs(fy_jac)/(2*self.fy_interp(mu)**2)).shape, (fx*np.abs(fy_jac)/(2*self.fy_interp(mu)**2)*term).shape)
                
                term2 = fx*np.abs(fy_jac)/(2*self.fy_interp(mu)**2)*term
                # term2 = np.sqrt(fx/(self.fy_interp(mu)))*term # upper-bound to the term2 above
                if self.trig_id:
                    # w_raw = self.c_w2*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4) + self.c_w3*np.abs(term2) # with triangle inequality
                    w_raw = self.c_w2*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4) + self.c_w3*np.abs(term) # with triangle inequality
                    # w_raw = 0*fx/self.fy_interp(mu) + 1*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4) + 100*np.abs(term2) # with triangle inequality --- with tol 1e-3
                    # w_raw = 0*fx/self.fy_interp(mu) + 1*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4) + 1000*np.abs(term2) # with triangle inequality --- with tol 1e-2
                else:
                    w_raw = 1*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4) - term2 + self.const_w # with triangle inequality
                    # print(self.const_w)
                
                # print(w_raw)
                
                # plt.figure()
                # plt.plot(w_raw[w_raw>.05])
                
                # w_raw = 0*fx/self.fy_interp(mu) + 1*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4) - 1*term2 + 1000 # without triangle inequality
                # print('lh w_raw = {}'.format(1*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4) - 1*term2))
                # print('lh w_raw + 100000 = {}'.format(w_raw))
                c0 = fx*np.abs(fy_jac)**2/(self.fy_interp(mu)**3)
                # print('c0 = {}'.format(c0))
                
            w_raw = np.nan_to_num(w_raw)
            c0 = np.nan_to_num(c0)
            self.gmm = self._fit_gmm(pts, w_raw, self.kwargs_gmm)
            self.c0_gmm = self._fit_gmm(pts, c0, self.kwargs_gmm)

        return self

    @staticmethod
    def _fit_gmm(pts, w_raw, kwargs_gmm):
        """Fit Gaussian Mixture Model using scikit's GMM framework.

        Parameters
        ----------
        pts : array
            Sample points. 
        w_raw : array
            Raw likelihood ratio at sample points.
        kwargs_gmm : dict
            A dictionary of keyword arguments for scikit's GMM routine.

        Returns
        -------
        gmm : scikit Gaussian Mixture Model
            A GMM object approximating the likelihood ratio.

        """
        # Sample and fit
        sca = np.sum(w_raw)
        rng = np.random.default_rng()
        try:
            aa = rng.choice(pts, size=20000, p=w_raw/sca)
        except:
            print('Fitting gmm failed because of a problem in the probability!!')
            print('non-positive prob.: ', w_raw/sca[(w_raw/sca)<=0])
            print('nan prob.: ', w_raw/sca[np.isnan(w_raw/sca)])
        gmm = GMM(**kwargs_gmm)
        gmm = gmm.fit(X=aa)
        # Rescale
        gmm_y = np.exp(gmm.score_samples(pts))
        scgmm = np.sum(gmm_y)
        gmm.weights_ *= (sca/w_raw.shape[0] * gmm_y.shape[0]/scgmm)
        return gmm

    @staticmethod
    def check_weight_type(weight_type):
        assert(weight_type.lower() in ["nominal", "importance", "importance_ho"])
        return weight_type.lower()



# %% Original likelihood func
# import numpy as np
# from scipy.interpolate import InterpolatedUnivariateSpline
# from sklearn.mixture import GaussianMixture as GMM
# from .utils import fix_dim_gmm, custom_KDE
# import numpy as np
# import matplotlib.pyplot as plt


# class Likelihood(object):
#     """A class for computation of the likelihood ratio.

#     Parameters
#     ----------
#     model : instance of GPRegression
#         A GPy model 
#     inputs : instance of Inputs
#         The input space.
#     weight_type : str, optional
#         Type of likelihood weight. Must be one of
#             - "nominal" : uses w(x) = p(x)
#             - "importance" : uses w(x) = p(x)/p_y(mu(x))
#             - "importance_H.O." # Alireza Mojahed
#     fit_gmm : boolean, optional
#         Whether or not to use a GMM approximation for the likelihood
#         ratio.  
#     kwargs_gmm : dict, optional
#         A dictionary of keyword arguments for scikit's GMM routine.
#         Use this to specify the number of Gaussian mixtures and the
#         type of covariance matrix.

#     Attributes
#     ----------
#     model, inputs, weight_type, fit_gmm, kwargs_gmm : see Parameters
#     fy_interp : scipy 1-D interpolant
#         An interpolant for the output pdf p_y(mu)
#     gmm : scikit Gaussian Mixture Model
#         A GMM object approximating the likelihood ratio.

#     """

#     def __init__(self, model, inputs, weight_type="importance", 
#                  fit_gmm=True, kwargs_gmm=None):

#         self.model = model
#         self.inputs = inputs
#         self.weight_type = self.check_weight_type(weight_type)
#         self.fit_gmm = fit_gmm

#         if kwargs_gmm is None:
#             kwargs_gmm = dict(n_components=2, covariance_type="full")
#         self.kwargs_gmm = kwargs_gmm

#         self._prepare_likelihood()

#     def update_model(self, model):
#         self.model = model
#         self._prepare_likelihood()
#         return self

#     def evaluate(self, x):
#         """Evaluates the likelihood ratio at x.

#         Parameters
#         ----------
#         x : array
#             Query points. Should be of size (n_pts, n_dim)

#         Returns
#         -------
#         w : array
#             The likelihood ratio at x.

#         """
#         if self.fit_gmm:
#             w = self._evaluate_gmm(x)
#         else:
#             w = self._evaluate_raw(x)
#         return w

#     def jacobian(self, x):
#         """Evaluates the gradients of the likelihood ratio at x.

#         Parameters
#         ----------
#         x : array
#             Query points. Should be of size (n_pts, n_dim)

#         Returns
#         -------
#         w_jac : array
#             Gradients of the likelihood ratio at x.

#         """
#         if self.fit_gmm:
#             w_jac = self._jacobian_gmm(x)
#         else:
#             w_jac = self._jacobian_raw(x)
#         return w_jac

#     def _evaluate_gmm(self, x):
#         x = np.atleast_2d(x)
#         w = np.exp(self.gmm.score_samples(x))
#         return w[:,None]

#     def _jacobian_gmm(self, x):
#         x = np.atleast_2d(x)
#         w_jac = np.zeros(x.shape)
#         p = np.exp(self.gmm._estimate_weighted_log_prob(x))
#         precisions = fix_dim_gmm(self.gmm, matrix_type="precisions")
#         for ii in range(self.gmm.n_components):
#             w_jac += p[:,ii,None] * np.dot(self.gmm.means_[ii]-x, \
#                                            precisions[ii])
#         return w_jac

#     def _evaluate_raw(self, x):
#         x = np.atleast_2d(x)
#         fx = self.inputs.pdf(x)
#         if self.weight_type == "nominal": 
#             w = fx
#         elif self.weight_type == "importance":
#             mu = self.model.predict(x)[0].flatten()
#             if self.model.normalizer:
#                 mu = self.model.normalizer.normalize(mu)
#             fy = self.fy_interp(mu)
#             w = fx/fy
#         # """Addition by Alireza Mojahed"""
#         elif self.weight_type == "importance_ho":
#             mu = self.model.predict(x)[0].flatten()
#             if self.model.normalizer:
#                 mu = self.model.normalizer.normalize(mu)
#             fy = self.fy_interp(mu)
#             w = fx/fy
#         return w[:,None]

#     def _jacobian_raw(self, x):
#         x = np.atleast_2d(x)
#         fx_jac = self.inputs.pdf_jac(x)

#         if self.weight_type == "nominal":
#             w_jac = fx_jac

#         elif self.weight_type == "importance":
#             mu = self.model.predict(x)[0].flatten()
#             if self.model.normalizer:
#                 mu = self.model.normalizer.normalize(mu)
#             mu_jac, _ = self.model.predictive_gradients(x)
#             mu_jac = mu_jac[:,:,0]
#             fx = self.inputs.pdf(x)
#             fy = self.fy_interp(mu)
#             fy_jac = self.fy_interp.derivative()(mu)
#             tmp = fx * fy_jac / fy**2
#             w_jac = fx_jac / fy[:,None] - tmp[:,None] * mu_jac
            
#         # """Addition by Alireza Mojahed"""            
#         elif self.weight_type == "importance_ho":
#             mu = self.model.predict(x)[0].flatten()
#             if self.model.normalizer:
#                 mu = self.model.normalizer.normalize(mu)
#             mu_jac, _ = self.model.predictive_gradients(x)
#             mu_jac = mu_jac[:,:,0]
#             fx = self.inputs.pdf(x)
#             fy = self.fy_interp(mu)
#             fy_jac = self.fy_interp.derivative()(mu)
#             tmp = fx * fy_jac / fy**2
#             w_jac = fx_jac / fy[:,None] - tmp[:,None] * mu_jac
        

#         return w_jac

#     def _prepare_likelihood(self):
#         """Prepare likelihood ratio for evaluation."""

#         if self.inputs.input_dim <= 2:
#             n_samples = int(1e5)
#         else: 
#             n_samples = int(1e6)

#         pts = self.inputs.draw_samples(n_samples=n_samples, 
#                                        sample_method="uni")
#         fx = self.inputs.pdf(pts)

#         if self.weight_type == "importance":
#             mu = self.model.predict(pts)[0].flatten()
#             if self.model.normalizer:
#                 mu = self.model.normalizer.normalize(mu)
#             x, y = custom_KDE(mu, weights=fx).evaluate()
#             self.fy_interp = InterpolatedUnivariateSpline(x, y, k=1)
            
#         # """Addition by Alireza Mojahed"""
#         if self.weight_type == "importance_ho":
#             mu = self.model.predict(pts)[0].flatten()
#             mu_grad = self.model.predictive_gradients(pts)[1] # Check shape
#             mu_hess = self.model.predictive_hessian(pts) # Check shape
#             if self.model.normalizer:
#                 mu = self.model.normalizer.normalize(mu)
#             x, y = custom_KDE(mu, weights=fx).evaluate()
#             self.fy_interp = InterpolatedUnivariateSpline(x, y, k=1)

#         if self.fit_gmm:
#             if self.weight_type == "nominal":
#                 c0 = 0
#                 w_raw = fx
#             elif self.weight_type == "importance":
#                 fy_jac = self.fy_interp.derivative()(mu)
#                 c0 = fx*np.abs(fy_jac)**2/(self.fy_interp(mu)**3)
#                 w_raw = fx/self.fy_interp(mu)
                
#                 # print(w_raw)
                
#                 # plt.figure()
#                 # plt.plot(w_raw[w_raw>.05],'r')
                
#             # """Addition by Alireza Mojahed"""
#             elif self.weight_type == "importance_ho":
#                 fy_jac = self.fy_interp.derivative()(mu)
#                 # term = []
#                 # for i in pts.shape[0]:
#                 #     d_mu = mu_grad[1][i,::]
#                 #     H_mu = mu_hess[i,::].reshape((pts.shape[1],pts.shape[1]))
#                 #     term[i] = d_mu.T@H_mu@d_mu/np.norm(mu_grad)**4
#                 term = np.empty(1)
#                 D_mu = np.empty((pts.shape[0],pts.shape[1]))
#                 Hess_mu = np.empty((pts.shape[0],pts.shape[1],pts.shape[1]))
#                 D_mu_norm4 = np.empty(pts.shape[0])
#                 D_mu__Hess_mu__D_mu = np.empty(pts.shape[0])
#                 for i in range(pts.shape[0]):
#                     d_mu = mu_grad[i,::]
#                     # D_mu = np.vstack((D_mu,d_mu))
#                     D_mu[i] = d_mu
#                     D_mu_norm4[i] = np.linalg.norm(d_mu)**4
#                     H_mu = mu_hess[i,::].reshape((pts.shape[1],pts.shape[1]))
#                     # Hess_mu = np.vstack((Hess_mu,H_mu.reshape((1,pts.shape[1],pts.shape[1]))))
#                     Hess_mu[i] = H_mu.reshape((1,pts.shape[1],pts.shape[1]))
#                     D_mu__Hess_mu__D_mu[i] = d_mu@H_mu@d_mu.T
#                     tol = 1e-5
#                     term = np.vstack((term,d_mu@H_mu@d_mu.T/(np.linalg.norm(d_mu)**4+tol)))
#                 # Hess_mu = Hess_mu[1:,::]
#                 # D_mu = D_mu[1:,::]
#                 term = term[1:].flatten()
                
#                 # print('D_mu is {}'.format(D_mu))
#                 # print('Hess_mu is {}'.format(Hess_mu))
#                 # print('|D_mu|^4 is {}'.format(D_mu_norm4))
#                 # print('D_mu*Hess_mu*D_mu is {}'.format(D_mu__Hess_mu__D_mu))
                
#                 term2 = fx*np.abs(fy_jac)/(2*self.fy_interp(mu)**2)*term
#                 w_raw = 0*fx/self.fy_interp(mu) + 1*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4) + 1*np.abs(term2) # with triangle inequality
                
#                 # print(w_raw)
                
#                 # plt.figure()
#                 # plt.plot(w_raw[w_raw>.05])
                
#                 # w_raw = 0*fx/self.fy_interp(mu) + 1*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4) - 1*term2 + 1000 # without triangle inequality
#                 # print('lh w_raw = {}'.format(1*fx**2*fy_jac**2/(2*self.fy_interp(mu)**4) - 1*term2))
#                 # print('lh w_raw + 100000 = {}'.format(w_raw))
#                 c0 = fx*np.abs(fy_jac)**2/(self.fy_interp(mu)**3)
#                 # print('c0 = {}'.format(c0))
                
                
#             self.gmm = self._fit_gmm(pts, w_raw, self.kwargs_gmm)
#             self.c0_gmm = self._fit_gmm(pts, c0, self.kwargs_gmm)

#         return self

#     @staticmethod
#     def _fit_gmm(pts, w_raw, kwargs_gmm):
#         """Fit Gaussian Mixture Model using scikit's GMM framework.

#         Parameters
#         ----------
#         pts : array
#             Sample points. 
#         w_raw : array
#             Raw likelihood ratio at sample points.
#         kwargs_gmm : dict
#             A dictionary of keyword arguments for scikit's GMM routine.

#         Returns
#         -------
#         gmm : scikit Gaussian Mixture Model
#             A GMM object approximating the likelihood ratio.

#         """
#         # Sample and fit
#         sca = np.sum(w_raw)
#         rng = np.random.default_rng()
#         aa = rng.choice(pts, size=20000, p=w_raw/sca)
#         gmm = GMM(**kwargs_gmm)
#         gmm = gmm.fit(X=aa)
#         # Rescale
#         gmm_y = np.exp(gmm.score_samples(pts))
#         scgmm = np.sum(gmm_y)
#         gmm.weights_ *= (sca/w_raw.shape[0] * gmm_y.shape[0]/scgmm)
#         return gmm

#     @staticmethod
#     def check_weight_type(weight_type):
#         assert(weight_type.lower() in ["nominal", "importance", "importance_ho"])
#         return weight_type.lower()


