import numpy as np
from ..acquisitions import *
from ..likelihood import Likelihood

class my_acq1_1():
    
     def __init__(self, model, inputs, trig_id=True, const_w=1e5, c_w2=1, c_w3=1, tol=1e-5):
        self.model = model
        self.inputs = inputs
        self.trig_id = trig_id
        self.const_w = const_w
        self.c_w2 = c_w2
        self.c_w3 = c_w3
        self.tol = tol
           
        likelihood1 = Likelihood(model, inputs, weight_type="importance")
        self.a1 = IVR_LW(model, inputs, likelihood=likelihood1)
        if self.trig_id:
            likelihood2 = Likelihood(model, inputs, weight_type="importance_ho", c_w2=self.c_w2, c_w3=self.c_w3, tol=self.tol)
            self.a2 = IVR_LW(model, inputs, likelihood=likelihood2)
        else:
            likelihood2 = Likelihood(model, inputs, weight_type="importance_ho", c_w2=self.c_w2, c_w3=self.c_w3, tol=self.tol, trig_id=False, const_w=self.const_w)
            self.a2 = IVR_LW_mod(model, inputs, likelihood=likelihood2, const_w=self.const_w)


     def evaluate(self, x):
        """Evaluates acquisition function at x."""
        # result = np.sqrt(64627.461142341046)*self.a1.evaluate(x) + 1*(self.a2.evaluate(x))**2
        # result = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) + 1*(self.a2.evaluate(x))**2 # Instead of the eq in the next it was this eq originally
        # print('acq w_raw = {}, a2.eval = {}'.format(result, self.a2.evaluate(x)-1000))
        result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 1 - successful (with trinagle inequality)
        # result = np.sqrt(self.a1.cauchy_schwarz_weight_squared(x))*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 3 - (with trinagle inequality)
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x)-1000)**2 # Candidate 1 - successful (without trinagle inequality)
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 2 - successful (with trinagle inequality)
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x)-1000)**2 # Candidate 2 - successful (without trinagle inequality)
        return result

     def jacobian(self, x):
        """Evaluates gradients of acquisition function at x."""
        # result_jac = np.sqrt(64627.461142341046)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x))
        # result_jac = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Instead of the eq in the next it was this eq originally
        result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 1 - successful (with trinagle inequality)
        # result_jac = np.sqrt(self.a1.cauchy_schwarz_weight_squared(x))*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 3 - (with trinagle inequality)
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)-1000) # Candidate 1 - successful (without trinagle inequality)
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 2 - successful (with trinagle inequality)
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)-1000) # Candidate 2 - successful (without trinagle inequality)
        return result_jac
   
     def update_parameters(self):
        self.a1.likelihood.update_model(self.model)
        self.a2.likelihood.update_model(self.model)
        
        

class my_acq2_1():
    
     def __init__(self, model, inputs, trig_id=True, const_w=1e5, c_w2=1, c_w3=1, tol=1e-5):
        self.model = model
        self.inputs = inputs
        self.trig_id = trig_id
        self.const_w = const_w
        self.c_w2 = c_w2
        self.c_w3 = c_w3
        self.tol = tol
           
        likelihood1 = Likelihood(model, inputs, weight_type="importance")
        self.a1 = IVR_LW(model, inputs, likelihood=likelihood1)
        if self.trig_id:
            likelihood2 = Likelihood(model, inputs, weight_type="importance_ho", c_w2=self.c_w2, c_w3=self.c_w3, tol=self.tol)
            self.a2 = IVR_LW(model, inputs, likelihood=likelihood2)
        else:
            likelihood2 = Likelihood(model, inputs, weight_type="importance_ho", c_w2=self.c_w2, c_w3=self.c_w3, tol=self.tol, trig_id=False, const_w=self.const_w)
            self.a2 = IVR_LW_mod(model, inputs, likelihood=likelihood2, const_w=self.const_w)


     def evaluate(self, x):
        """Evaluates acquisition function at x."""
        # result = np.sqrt(64627.461142341046)*self.a1.evaluate(x) + 1*(self.a2.evaluate(x))**2
        # result = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) + 1*(self.a2.evaluate(x))**2 # Instead of the eq in the next it was this eq originally
        # print('acq w_raw = {}, a2.eval = {}'.format(result, self.a2.evaluate(x)-1000))
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 1 - successful (with trinagle inequality)
        # result = np.sqrt(self.a1.cauchy_schwarz_weight_squared(x))*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 3 - (with trinagle inequality)
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x)-1000)**2 # Candidate 1 - successful (without trinagle inequality)
        result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 2 - successful (with trinagle inequality)
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x)-1000)**2 # Candidate 2 - successful (without trinagle inequality)
        return result

     def jacobian(self, x):
        """Evaluates gradients of acquisition function at x."""
        # result_jac = np.sqrt(64627.461142341046)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x))
        # result_jac = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Instead of the eq in the next it was this eq originally
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 1 - successful (with trinagle inequality)
        # result_jac = np.sqrt(self.a1.cauchy_schwarz_weight_squared(x))*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 3 - (with trinagle inequality)
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)-1000) # Candidate 1 - successful (without trinagle inequality)
        result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 2 - successful (with trinagle inequality)
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)-1000) # Candidate 2 - successful (without trinagle inequality)
        return result_jac
   
     def update_parameters(self):
        self.a1.likelihood.update_model(self.model)
        self.a2.likelihood.update_model(self.model)
        
        
class my_acq1_0():
    
     def __init__(self, model, inputs, trig_id=True, const_w=1e5, c_w2=1, c_w3=1, tol=1e-5):
        self.model = model
        self.inputs = inputs
        self.trig_id = trig_id
        self.const_w = const_w
        self.c_w2 = c_w2
        self.c_w3 = c_w3
        self.tol = tol
           
        likelihood1 = Likelihood(model, inputs, weight_type="importance")
        self.a1 = IVR_LW(model, inputs, likelihood=likelihood1)
        if self.trig_id:
            likelihood2 = Likelihood(model, inputs, weight_type="importance_ho", c_w2=self.c_w2, c_w3=self.c_w3, tol=self.tol)
            self.a2 = IVR_LW(model, inputs, likelihood=likelihood2)
        else:
            likelihood2 = Likelihood(model, inputs, weight_type="importance_ho", c_w2=self.c_w2, c_w3=self.c_w3, tol=self.tol, trig_id=False, const_w=self.const_w)
            self.a2 = IVR_LW_mod(model, inputs, likelihood=likelihood2, const_w=self.const_w)


     def evaluate(self, x):
        """Evaluates acquisition function at x."""
        # result = np.sqrt(64627.461142341046)*self.a1.evaluate(x) + 1*(self.a2.evaluate(x))**2
        # result = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) + 1*(self.a2.evaluate(x))**2 # Instead of the eq in the next it was this eq originally
        # print('acq w_raw = {}, a2.eval = {}'.format(result, self.a2.evaluate(x)-1000))
        result = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 1 - successful (with trinagle inequality)
        # result = np.sqrt(self.a1.cauchy_schwarz_weight_squared(x))*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 3 - (with trinagle inequality)
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x)-1000)**2 # Candidate 1 - successful (without trinagle inequality)
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 2 - successful (with trinagle inequality)
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x)-1000)**2 # Candidate 2 - successful (without trinagle inequality)
        return result

     def jacobian(self, x):
        """Evaluates gradients of acquisition function at x."""
        # result_jac = np.sqrt(64627.461142341046)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x))
        # result_jac = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Instead of the eq in the next it was this eq originally
        result_jac = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 1 - successful (with trinagle inequality)
        # result_jac = np.sqrt(self.a1.cauchy_schwarz_weight_squared(x))*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 3 - (with trinagle inequality)
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)-1000) # Candidate 1 - successful (without trinagle inequality)
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 2 - successful (with trinagle inequality)
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)-1000) # Candidate 2 - successful (without trinagle inequality)
        return result_jac
   
     def update_parameters(self):
        self.a1.likelihood.update_model(self.model)
        self.a2.likelihood.update_model(self.model)
        
        

class my_acq2_0():
    
     def __init__(self, model, inputs, trig_id=True, const_w=1e5, c_w2=1, c_w3=1, tol=1e-5):
        self.model = model
        self.inputs = inputs
        self.trig_id = trig_id
        self.const_w = const_w
        self.c_w2 = c_w2
        self.c_w3 = c_w3
        self.tol = tol
           
        likelihood1 = Likelihood(model, inputs, weight_type="importance")
        self.a1 = IVR_LW(model, inputs, likelihood=likelihood1)
        if self.trig_id:
            likelihood2 = Likelihood(model, inputs, weight_type="importance_ho", c_w2=self.c_w2, c_w3=self.c_w3, tol=self.tol)
            self.a2 = IVR_LW(model, inputs, likelihood=likelihood2)
        else:
            likelihood2 = Likelihood(model, inputs, weight_type="importance_ho", c_w2=self.c_w2, c_w3=self.c_w3, tol=self.tol, trig_id=False, const_w=self.const_w)
            self.a2 = IVR_LW_mod(model, inputs, likelihood=likelihood2, const_w=self.const_w)


     def evaluate(self, x):
        """Evaluates acquisition function at x."""
        # result = np.sqrt(64627.461142341046)*self.a1.evaluate(x) + 1*(self.a2.evaluate(x))**2
        # result = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) + 1*(self.a2.evaluate(x))**2 # Instead of the eq in the next it was this eq originally
        # print('acq w_raw = {}, a2.eval = {}'.format(result, self.a2.evaluate(x)-1000))
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 1 - successful (with trinagle inequality)
        # result = np.sqrt(self.a1.cauchy_schwarz_weight_squared(x))*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 3 - (with trinagle inequality)
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x)-1000)**2 # Candidate 1 - successful (without trinagle inequality)
        result = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x))**2 # Candidate 2 - successful (with trinagle inequality)
        # result = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.evaluate(x) - 1*(self.a2.evaluate(x)-1000)**2 # Candidate 2 - successful (without trinagle inequality)
        return result

     def jacobian(self, x):
        """Evaluates gradients of acquisition function at x."""
        # result_jac = np.sqrt(64627.461142341046)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x))
        # result_jac = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Instead of the eq in the next it was this eq originally
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 1 - successful (with trinagle inequality)
        # result_jac = np.sqrt(self.a1.cauchy_schwarz_weight_squared(x))*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 3 - (with trinagle inequality)
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) - 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)-1000) # Candidate 1 - successful (without trinagle inequality)
        result_jac = 0*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)) # Candidate 2 - successful (with trinagle inequality)
        # result_jac = 1*self.a1.cauchy_schwarz_weight_squared(x)*self.a1.jacobian(x) + 1*2*(self.a2.jacobian(x))*(self.a2.evaluate(x)-1000) # Candidate 2 - successful (without trinagle inequality)
        return result_jac
   
     def update_parameters(self):
        self.a1.likelihood.update_model(self.model)
        self.a2.likelihood.update_model(self.model)
        
        
