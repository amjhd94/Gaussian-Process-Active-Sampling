import numpy as np
import GPy


class MyModel(GPy.models.GPRegression):

    def predictive_hessian(self, Xnew):

        mean_hes = np.empty((Xnew.shape[0], Xnew.shape[1], 
                             Xnew.shape[1], self.output_dim))

        for i in range(self.output_dim):
            one = np.ones(self._predictive_variable.shape[0])
            dK2_dX2 = self.kern.gradients_XX(one, Xnew, self._predictive_variable)
            wood = self.posterior.woodbury_vector[:,i:i+1].T
            hes = -np.sum( dK2_dX2*wood[...,None,None], axis=1 )
            mean_hes[:, :, :, i] = 0.5 * ( hes + hes.transpose(0,2,1) )

        return mean_hes


np.random.seed(2)

# Random dataset
N, M, Q = 2, 15, 3
X = np.random.rand(M,Q)
Y = np.random.rand(M,1)
x = np.random.rand(N,Q)

# Custom model
normalize_Y = False
ker = GPy.kern.RBF(input_dim=X.shape[1])
model = MyModel(X=X, Y=Y, kernel=ker, normalizer=normalize_Y)

# Quick & dirty finite-difference approximation for Hessian
eps = 1e-5
mu_hes_num = np.zeros( (x.shape[0], x.shape[1], x.shape[1]) )
for nn in range(x.shape[0]):
    xn = x[nn:nn+1]
    for ii in range(x.shape[1]):
        eps_ii = eps*np.eye(1, x.shape[1], ii)
        for jj in range(x.shape[1]):
            eps_jj = eps*np.eye(1, x.shape[1], jj)
            mu_ii_jj, _ = model.predict(xn + eps_ii + eps_jj)
            mu_ii, _ = model.predict(xn + eps_ii)
            mu_jj, _ = model.predict(xn + eps_jj)
            mu, _ = model.predict(xn)
            hes = (mu_ii_jj-mu_ii-mu_jj+mu)/eps**2
            mu_hes_num[nn,ii,jj] = (hes+hes.T)*0.5

# Hessian from custom class
mu_hes = model.predictive_hessian(x)

print("HESSIAN FD \n", mu_hes_num) 
print("HESSIAN GP \n", mu_hes[...,0])
assert(np.allclose(mu_hes_num, mu_hes[...,0], rtol=eps, atol=eps))



