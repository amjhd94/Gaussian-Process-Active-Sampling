import numpy as np
import GPy
# from gpsearch.core.kern_with_new_gradientX2 import kern_with_gradients_X2
# from .kernels import *

class MyModelWithHessian(GPy.models.GPRegression):


    def predictive_hessian(self, Xnew, kern=None):
        
        if kern is None:
            kern = self.kern

        mean_hes = np.empty((Xnew.shape[0],Xnew.shape[1], Xnew.shape[1], self.output_dim))

        for i in range(self.output_dim):
            one = np.ones(self._predictive_variable.shape[0])
            dK2_dX2 = self.kern.gradients_XX(one, Xnew, self._predictive_variable)
            wood = self.posterior.woodbury_vector[:,i:i+1].T
            hes = -np.sum( dK2_dX2*wood[...,None,None], axis=1 )
            mean_hes[:, :, :, i] = 0.5 * ( hes + hes.transpose(0,2,1) )

        if self.normalizer is not None:
            mean_hes = self.normalizer.inverse_mean(mean_hes) \
                       - self.normalizer.inverse_mean(0.)

        return mean_hes
    
    def predictive_grad_mine(self, Xnew, kern=None):
        
        if kern is None:
            kern = self.kern
        # fix the output shape
        mean_jac = np.empty((Xnew.shape[0],Xnew.shape[0], self.output_dim))
        

        for i in range(self.output_dim):
            mean_jac[:, :, i] = kern.gradients_X(
                self.posterior.woodbury_vector[:, i:i+1].T, Xnew,
                self._predictive_variable)        

        
        if self.normalizer is not None:
            mean_jac = self.normalizer.inverse_mean(mean_jac) \
                       - self.normalizer.inverse_mean(0.)
            

        return mean_jac
    
    
    
########### Need to define my own gradients_XX
    def gradients_X2(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective K(dL_dK), compute the second derivative of K wrt X and X2:
        returns the full covariance matrix [QxQ] of the input dimensionfor each pair or vectors, thus
        the returned array is of shape [NxNxQxQ].
        ..math:
            \frac{\partial^2 K}{\partial X ^2} = - \frac{\partial^2 K}{\partial X^2}
        ..returns:
            dL2_dX2:  [NxMxQxQ] in the cov=True case, or [NxMxQ] in the cov=False case,
                        for X [NxQ] and X2[MxQ] (X2 is X if, X2 is None)
                        Thus, we return the second derivative in X2.
        """
        # According to multivariable chain rule, we can chain the second derivative through r:
        # d2K_dXdX2 = dK_dr*d2r_dXdX2 + d2K_drdr * dr_dX * dr_dX2:
        # d2K_dX^2 = dK_dr*d2r_dX^2 + d2K_drdr * (dr_dX)^2:
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
        # grad = (((dL_dK*(tmp1*invdist2 - tmp2))[:,:,None,None] * dist)/l2[None,None,:,None]
        #         - (dL_dK*tmp1)[:,:,None,None] * I)/l2[None,None,None,:]
        grad = ((dL_dK*(tmp1*invdist2 - tmp2))[:,:,None,None] * dist)/l2[None,None,:,None]/l2[None,None,None,:] - (dL_dK*tmp1)[:,:,None,None] * I/l2[None,None,None,:]
        return grad