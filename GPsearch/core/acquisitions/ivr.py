import numpy as np
from ..utils import fix_dim_gmm
from .base import Acquisition, AcquisitionWeighted
import matplotlib.pyplot as plt


class IVR(Acquisition):
    """A class for Integrated Variance Reduction.

    Parameters
    ----------
    model, inputs : see parent class (Acquisition)

    Attributes
    ----------
    model, inputs : see Parameters

    """

    def evaluate(self, x):
        x = np.atleast_2d(x)
        _, var = self.model.predict_noiseless(x) # Noiseless ghost point
       #_, var = self.model.predict(x) 
        integral = self.integrate_covariance(x)
        if self.model.normalizer:
            var /= self.model.normalizer.std**2
        
        # print('norm intergral min is ', np.min(integral/np.max(integral)))
        # print('var min is ', np.min(var))
        
        ivr = integral / var # Original
        
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.plot(integral/np.max(integral))
        # plt.plot(var/np.max(var))
        # plt.subplot(1,2,2)
        # plt.semilogy(integral/np.max(integral))
        # plt.semilogy(var/np.max(var))
        
        # ivr = integral
        
        return -ivr

    def jacobian(self, x):
        x = np.atleast_2d(x)
        _, var = self.model.predict_noiseless(x) # Noiseless ghost point
       #_, var = self.model.predict(x) 
        _, var_jac = self.model.predictive_gradients(x)
        integral = self.integrate_covariance(x)
        integral_jac = self.integrate_covariance_jacobian(x)
        if self.model.normalizer:
            var /= self.model.normalizer.std**2
        ivr_jac = (integral_jac*var - integral*var_jac) / var**2 # Original
        # ivr_jac = (integral_jac*var) / var**2
        return -ivr_jac

    def integrate_covariance(self, x):
        """Compute \int cov(x,x')^2 dx'."""
        K = self.model.kern
        X = self.model.X
        Skk_inv = self.model.posterior.woodbury_inv
        y_k = np.dot(Skk_inv, K.K(X,x))

        term1 = K.IntKK(x) 
        term2 = np.dot(K.IntKK(X), y_k)
        term3 = K.IntKK(X,x)
        int_cov = term1 + np.dot(y_k.T, term2 - 2*term3)
        int_cov = np.diag(int_cov)[:,None]
        return int_cov
 
    def integrate_covariance_jacobian(self, x):
        """Compute d/dx \int cov(x,x')^2 dx'."""
        K = self.model.kern
        X = self.model.X
        Skk_inv = self.model.posterior.woodbury_inv
        jac_ker = -K.gradients_X(np.ones((1,x.shape[0])), X, x) # Beware minus sign!
        y_k = np.dot(Skk_inv, K.K(X,x))
        jac_y_k = np.dot(Skk_inv, jac_ker)

        dterm1_dX = 2*K.dIntKK_dX(x)
        dterm2_dX = np.dot(K.IntKK(X), jac_y_k)
        dterm3_dX = K.dIntKK_dX(x,X)

        int_jac = dterm1_dX + 2*np.dot(y_k.T, dterm2_dX) \
                  - 2*np.dot(y_k.T, dterm3_dX) \
                  - 2*np.dot(K.IntKK(X,x).T, jac_y_k)
        return int_jac



class IVR_LW(AcquisitionWeighted, IVR):
    """A class for Likelihood-Weighted Integrated Variance Reduction.

    Parameters
    ----------
    model, inputs : see parent class (AcquisitionWeighted)

    Attributes
    ----------
    model, inputs : see Parameters

    Notes
    -----
    This subclass overrides `integrate_covariance` and 
    `integrate_covariance_jacobian` of the `IVR` class.
    
    """

    def integrate_covariance(self, x):
        """Compute \int cov(x,x')^2 w_gmm(x') dx'."""
        K = self.model.kern
        X = self.model.X
        Skk_inv = self.model.posterior.woodbury_inv
        y_k = np.dot(Skk_inv, K.K(X,x))

        int_cov = 0.0
        gmm = self.likelihood.gmm
        covs = fix_dim_gmm(gmm, matrix_type="covariance")

        for ii in range(gmm.n_components):

            mu_i = gmm.means_[ii]
            cov_i = covs[ii]
            wei = gmm.weights_[ii]

            term1 = K.IntKKNorm(x, x, mu_i, cov_i)
            term2 = np.dot( K.IntKKNorm(X, X, mu_i, cov_i), y_k )
            term3 = K.IntKKNorm(X, x, mu_i, cov_i)
            tmp_i = term1 + np.dot(y_k.T, term2 - 2*term3) 

            int_cov += wei*tmp_i

        int_cov = np.diag(int_cov)[:,None]
        return int_cov

    def integrate_covariance_jacobian(self, x):
        """Compute d/dx \int cov(x,x')^2 w_gmm(x') dx'."""
        K = self.model.kern
        X = self.model.X
        Skk_inv = self.model.posterior.woodbury_inv
        jac_ker = -K.gradients_X(np.ones((1,x.shape[0])), X, x) # Beware minus sign!
        y_k = np.dot(Skk_inv, K.K(X,x))
        jac_y_k = np.dot(Skk_inv, jac_ker)

        int_jac = 0.0
        gmm = self.likelihood.gmm
        covs = fix_dim_gmm(gmm, matrix_type="covariance")

        for ii in range(gmm.n_components):
            mu_i = gmm.means_[ii]
            cov_i = covs[ii]
            wei = gmm.weights_[ii]

            dterm1_dX = 2*K.dIntKKNorm_dX(x, x, mu_i, cov_i)
            dterm2_dX = np.dot( K.IntKKNorm(X,X,mu_i,cov_i), jac_y_k)
            dterm3_dX = K.dIntKKNorm_dX(x, X, mu_i, cov_i)

            tmp_i = dterm1_dX + 2*np.dot(y_k.T, dterm2_dX) \
                    - 2*np.dot(y_k.T, dterm3_dX) \
                    - 2*np.dot(K.IntKKNorm(X,x,mu_i,cov_i).T, jac_y_k)

            int_jac += wei*tmp_i

        return int_jac
    
    def cauchy_schwarz_weight_squared(self, x):
        """Compute (\int Px*|Py'|^2/Py^3 dx)^2."""
        int_c0 = 0.0
        c0_gmm = self.likelihood.c0_gmm
        covs = fix_dim_gmm(c0_gmm, matrix_type="covariance")

        for ii in range(c0_gmm.n_components):

            
            wei = c0_gmm.weights_[ii]

            tmp_i = np.array([[1]])

            int_c0 += wei*tmp_i

        int_c0 = np.diag(int_c0)[:,None]
        # return int_c0**2 # 1st try
        return int_c0 # 2nd try
    
    
    
class IVR_LW_mod(IVR_LW):
    """A class for Likelihood-Weighted Integrated Variance Reduction.

    Parameters
    ----------
    model, inputs : see parent class (AcquisitionWeighted)

    Attributes
    ----------
    model, inputs : see Parameters

    Notes
    -----
    This subclass overrides `integrate_covariance` and 
    `integrate_covariance_jacobian` of the `IVR` class.
    
    """
    
    def __init__(self, model, inputs, likelihood, const_w):
        super().__init__(model, inputs, likelihood=None)
        self.const_w = const_w
    
    def integrate_covariance(self, x):
        """Compute \int cov(x,x')^2 w_gmm(x') dx'."""
        K = self.model.kern
        X = self.model.X
        Skk_inv = self.model.posterior.woodbury_inv
        y_k = np.dot(Skk_inv, K.K(X,x))

        int_cov = 0.0
        gmm = self.likelihood.gmm
        covs = fix_dim_gmm(gmm, matrix_type="covariance")

        for ii in range(gmm.n_components+1):
            if ii==0:
                # const_w = 1e5
                wei = self.const_w
                # print(self.const_w)
                term1 = K.IntKK(x, x)
                term2 = np.dot( K.IntKK(X, X), y_k )
                term3 = K.IntKK(X, x)
                tmp_i = term1 + np.dot(y_k.T, term2 - 2*term3) 
    
                int_cov += -wei*tmp_i
                
            else:            
                mu_i = gmm.means_[ii-1]
                cov_i = covs[ii-1]
                wei = gmm.weights_[ii-1]
    
                term1 = K.IntKKNorm(x, x, mu_i, cov_i)
                term2 = np.dot( K.IntKKNorm(X, X, mu_i, cov_i), y_k )
                term3 = K.IntKKNorm(X, x, mu_i, cov_i)
                tmp_i = term1 + np.dot(y_k.T, term2 - 2*term3) 
    
                int_cov += wei*tmp_i

        int_cov = np.diag(int_cov)[:,None]
        return int_cov

    def integrate_covariance_jacobian(self, x):
        """Compute d/dx \int cov(x,x')^2 w_gmm(x') dx'."""
        K = self.model.kern
        X = self.model.X
        Skk_inv = self.model.posterior.woodbury_inv
        jac_ker = -K.gradients_X(np.ones((1,x.shape[0])), X, x) # Beware minus sign!
        y_k = np.dot(Skk_inv, K.K(X,x))
        jac_y_k = np.dot(Skk_inv, jac_ker)

        int_jac = 0.0
        gmm = self.likelihood.gmm
        covs = fix_dim_gmm(gmm, matrix_type="covariance")

        for ii in range(gmm.n_components+1):
            if ii==0:
                wei = self.const_w
    
                dterm1_dX = 2*K.dIntKK_dX(x, x)
                dterm2_dX = np.dot( K.IntKK(X,X), jac_y_k)
                dterm3_dX = K.dIntKK_dX(x, X)
    
                tmp_i = dterm1_dX + 2*np.dot(y_k.T, dterm2_dX) \
                        - 2*np.dot(y_k.T, dterm3_dX) \
                        - 2*np.dot(K.IntKK(X,x).T, jac_y_k)
    
                int_jac += -wei*tmp_i
                
            else:
            
                mu_i = gmm.means_[ii-1]
                cov_i = covs[ii-1]
                wei = gmm.weights_[ii-1]
    
                dterm1_dX = 2*K.dIntKKNorm_dX(x, x, mu_i, cov_i)
                dterm2_dX = np.dot( K.IntKKNorm(X,X,mu_i,cov_i), jac_y_k)
                dterm3_dX = K.dIntKKNorm_dX(x, X, mu_i, cov_i)
    
                tmp_i = dterm1_dX + 2*np.dot(y_k.T, dterm2_dX) \
                        - 2*np.dot(y_k.T, dterm3_dX) \
                        - 2*np.dot(K.IntKKNorm(X,x,mu_i,cov_i).T, jac_y_k)
    
                int_jac += wei*tmp_i

        return int_jac


# %% Original ivr
# import numpy as np
# from ..utils import fix_dim_gmm
# from .base import Acquisition, AcquisitionWeighted


# class IVR(Acquisition):
#     """A class for Integrated Variance Reduction.

#     Parameters
#     ----------
#     model, inputs : see parent class (Acquisition)

#     Attributes
#     ----------
#     model, inputs : see Parameters

#     """

#     def evaluate(self, x):
#         x = np.atleast_2d(x)
#         _, var = self.model.predict_noiseless(x) # Noiseless ghost point
#        #_, var = self.model.predict(x) 
#         integral = self.integrate_covariance(x)
#         if self.model.normalizer:
#             var /= self.model.normalizer.std**2
#         ivr = integral / var
#         return -ivr

#     def jacobian(self, x):
#         x = np.atleast_2d(x)
#         _, var = self.model.predict_noiseless(x) # Noiseless ghost point
#        #_, var = self.model.predict(x) 
#         _, var_jac = self.model.predictive_gradients(x)
#         integral = self.integrate_covariance(x)
#         integral_jac = self.integrate_covariance_jacobian(x)
#         if self.model.normalizer:
#             var /= self.model.normalizer.std**2
#         ivr_jac = (integral_jac*var - integral*var_jac) / var**2
#         return -ivr_jac

#     def integrate_covariance(self, x):
#         """Compute \int cov(x,x')^2 dx'."""
#         K = self.model.kern
#         X = self.model.X
#         Skk_inv = self.model.posterior.woodbury_inv
#         y_k = np.dot(Skk_inv, K.K(X,x))

#         term1 = K.IntKK(x) 
#         term2 = np.dot(K.IntKK(X), y_k)
#         term3 = K.IntKK(X,x)
#         int_cov = term1 + np.dot(y_k.T, term2 - 2*term3)
#         int_cov = np.diag(int_cov)[:,None]
#         return int_cov
 
#     def integrate_covariance_jacobian(self, x):
#         """Compute d/dx \int cov(x,x')^2 dx'."""
#         K = self.model.kern
#         X = self.model.X
#         Skk_inv = self.model.posterior.woodbury_inv
#         jac_ker = -K.gradients_X(np.ones((1,x.shape[0])), X, x) # Beware minus sign!
#         y_k = np.dot(Skk_inv, K.K(X,x))
#         jac_y_k = np.dot(Skk_inv, jac_ker)

#         dterm1_dX = 2*K.dIntKK_dX(x)
#         dterm2_dX = np.dot(K.IntKK(X), jac_y_k)
#         dterm3_dX = K.dIntKK_dX(x,X)

#         int_jac = dterm1_dX + 2*np.dot(y_k.T, dterm2_dX) \
#                   - 2*np.dot(y_k.T, dterm3_dX) \
#                   - 2*np.dot(K.IntKK(X,x).T, jac_y_k)
#         return int_jac



# class IVR_LW(AcquisitionWeighted, IVR):
#     """A class for Likelihood-Weighted Integrated Variance Reduction.

#     Parameters
#     ----------
#     model, inputs : see parent class (AcquisitionWeighted)

#     Attributes
#     ----------
#     model, inputs : see Parameters

#     Notes
#     -----
#     This subclass overrides `integrate_covariance` and 
#     `integrate_covariance_jacobian` of the `IVR` class.
    
#     """

#     def integrate_covariance(self, x):
#         """Compute \int cov(x,x')^2 w_gmm(x') dx'."""
#         K = self.model.kern
#         X = self.model.X
#         Skk_inv = self.model.posterior.woodbury_inv
#         y_k = np.dot(Skk_inv, K.K(X,x))

#         int_cov = 0.0
#         gmm = self.likelihood.gmm
#         covs = fix_dim_gmm(gmm, matrix_type="covariance")

#         for ii in range(gmm.n_components):

#             mu_i = gmm.means_[ii]
#             cov_i = covs[ii]
#             wei = gmm.weights_[ii]

#             term1 = K.IntKKNorm(x, x, mu_i, cov_i)
#             term2 = np.dot( K.IntKKNorm(X, X, mu_i, cov_i), y_k )
#             term3 = K.IntKKNorm(X, x, mu_i, cov_i)
#             tmp_i = term1 + np.dot(y_k.T, term2 - 2*term3) 

#             int_cov += wei*tmp_i

#         int_cov = np.diag(int_cov)[:,None]
#         return int_cov

#     def integrate_covariance_jacobian(self, x):
#         """Compute d/dx \int cov(x,x')^2 w_gmm(x') dx'."""
#         K = self.model.kern
#         X = self.model.X
#         Skk_inv = self.model.posterior.woodbury_inv
#         jac_ker = -K.gradients_X(np.ones((1,x.shape[0])), X, x) # Beware minus sign!
#         y_k = np.dot(Skk_inv, K.K(X,x))
#         jac_y_k = np.dot(Skk_inv, jac_ker)

#         int_jac = 0.0
#         gmm = self.likelihood.gmm
#         covs = fix_dim_gmm(gmm, matrix_type="covariance")

#         for ii in range(gmm.n_components):
#             mu_i = gmm.means_[ii]
#             cov_i = covs[ii]
#             wei = gmm.weights_[ii]

#             dterm1_dX = 2*K.dIntKKNorm_dX(x, x, mu_i, cov_i)
#             dterm2_dX = np.dot( K.IntKKNorm(X,X,mu_i,cov_i), jac_y_k)
#             dterm3_dX = K.dIntKKNorm_dX(x, X, mu_i, cov_i)

#             tmp_i = dterm1_dX + 2*np.dot(y_k.T, dterm2_dX) \
#                     - 2*np.dot(y_k.T, dterm3_dX) \
#                     - 2*np.dot(K.IntKKNorm(X,x,mu_i,cov_i).T, jac_y_k)

#             int_jac += wei*tmp_i

#         return int_jac
    
#     def cauchy_schwarz_weight_squared(self, x):
#         """Compute (\int Px*|Py'|^2/Py^3 dx)^2."""
#         int_c0 = 0.0
#         c0_gmm = self.likelihood.c0_gmm
#         covs = fix_dim_gmm(c0_gmm, matrix_type="covariance")

#         for ii in range(c0_gmm.n_components):

            
#             wei = c0_gmm.weights_[ii]

#             tmp_i = np.array([[1]])

#             int_c0 += wei*tmp_i

#         int_c0 = np.diag(int_c0)[:,None]
#         # return int_c0**2 # 1st try
#         return int_c0 # 2nd try


