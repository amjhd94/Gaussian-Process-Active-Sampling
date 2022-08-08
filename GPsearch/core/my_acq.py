import numpy as np
from ..acquisitions import *
from ..likelihood import Likelihood

class my_acq():
    
     def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs
        self.iter_counter = 0


        likelihood1 = Likelihood(model, inputs, weight_type="importance")
        likelihood2 = Likelihood(model, inputs, weight_type="importance_ho")
        self.a1 = IVR_LW(model, inputs, likelihood=likelihood1)
        self.a2 = IVR_LW(model, inputs, likelihood=likelihood2)


     def evaluate(self, x):
        """Evaluates acquisition function at x."""
        result = np.sqrt(self.a1.evaluate(x)) + self.a2.evaluate(x)
        return result

     def jacobian(self, x):
        """Evaluates gradients of acquisition function at x."""
        result_jac = self.a1.jacobian(x)/(2*np.sqrt(self.a1.evaluate(x))) + self.a2.jacobian(x)
        return result_jac
   
     def update_parameters(self):
        self.a1.likelihood.update_model(self.model)
        self.a2.likelihood.update_model(self.model)

    