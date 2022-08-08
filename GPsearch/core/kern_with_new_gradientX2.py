import numpy as np
import GPy
from scipy import integrate
from .kern import Kern
from ...core.parameterization import Param
from ...util.linalg import tdot
from ... import util
from ...util.config import config # for assesing whether to use cython
from paramz.caching import Cache_this
from paramz.transformations import Logexp

try:
    from . import stationary_cython
    use_stationary_cython = config.getboolean('cython', 'working')
except ImportError:
    print('warning in stationary: failed to import cython module: falling back to numpy')
    use_stationary_cython = False
# from .kernels import *

# class kern_with_gradients_X2(GPy.kern.src.stationary.Stationary):
class kern_with_gradients_X2(Kern):
    
    def My_gradients_X2(self, dL_dK, X, X2=None):
        
        # # According to multivariable chain rule, we can chain the second derivative through r:
        # # d2K_dXdX2 = dK_dr*d2r_dXdX2 + d2K_drdr * dr_dX * dr_dX2:
        # # d2K_dX^2 = dK_dr*d2r_dX^2 + d2K_drdr * (dr_dX)^2:
            
        # invdist = self._inv_dist(X, X2)
        # invdist2 = invdist**2
        # dL_dr = self.dK_dr_via_X(X, X2) #* dL_dK # we perform this product later
        # tmp1 = dL_dr * invdist
        # dL_drdr = self.dK2_drdr_via_X(X, X2) #* dL_dK # we perofrm this product later
        # tmp2 = dL_drdr*invdist2
        # l2 =  np.ones(X.shape[1])*self.lengthscale**2 #np.multiply(np.ones(X.shape[1]) ,self.lengthscale**2)

        # if X2 is None:
        #     X2 = X
        #     tmp1 -= np.eye(X.shape[0])*self.variance
        # else:
        #     tmp1[invdist2==0.] -= self.variance

        # #grad = np.empty((X.shape[0], X2.shape[0], X2.shape[1], X.shape[1]), dtype=np.float64)
        # dist = X[:,None,:] - X2[None,:,:]
        # dist = (dist[:,:,:,None]*dist[:,:,None,:])
        # I = np.ones((X.shape[0], X2.shape[0], X2.shape[1], X.shape[1]))*np.eye((X2.shape[1]))
        # grad = (((dL_dK*(tmp1*invdist2 - tmp2))[:,:,None,None] * dist)/l2[None,None,:,None]
        #         - (dL_dK*tmp1)[:,:,None,None] * I)/l2[None,None,None,:]
        # # grad = ((dL_dK*(tmp1*invdist2 - tmp2))[:,:,None,None] * dist)/l2[None,None,:,None]/l2[None,None,None,:] - (dL_dK*tmp1)[:,:,None,None] * I/l2[None,None,None,:]
        # return grad 
        # According to multivariable chain rule, we can chain the second derivative through r:
        # d2K_dXdX2 = dK_dr*d2r_dXdX2 + d2K_drdr * dr_dX * dr_dX2:
        invdist = self._inv_dist(X, X2)
        invdist2 = invdist**2
        dL_dr = self.dK_dr_via_X(X, X2) #* dL_dK # we perform this product later
        tmp1 = dL_dr * invdist
        dL_drdr = self.dK2_drdr_via_X(X, X2) #* dL_dK # we perofrm this product later
        tmp2 = dL_drdr*invdist2
        l2 =  np.ones(X.shape[1])*self.lengthscale**2 #np.multiply(np.ones(X.shape[1]) ,self.lengthscale**2)

        if X2 is None:
            X2 = X
            tmp1 -= np.eye(X.shape[0])*self.variance
        else:
            tmp1[invdist2==0.] -= self.variance

        #grad = np.empty((X.shape[0], X2.shape[0], X2.shape[1], X.shape[1]), dtype=np.float64)
        dist = X[:,None,:] - X2[None,:,:]
        dist = (dist[:,:,:,None]*dist[:,:,None,:])
        I = np.ones((X.shape[0], X2.shape[0], X2.shape[1], X.shape[1]))*np.eye((X2.shape[1]))
        grad = (((dL_dK*(tmp1*invdist2 - tmp2))[:,:,None,None] * dist)/l2[None,None,:,None]
                - (dL_dK*tmp1)[:,:,None,None] * I)/l2[None,None,None,:]
        return grad
    
    


   