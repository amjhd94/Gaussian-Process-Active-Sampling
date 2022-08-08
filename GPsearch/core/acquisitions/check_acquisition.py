from ..acquisitions import *
from ..likelihood import Likelihood
import matplotlib.pyplot as plt


def check_acquisition(acquisition, model, inputs):

    if isinstance(acquisition, str):
        acq_name = acquisition.lower()

        if acq_name == "pi":
            return PI(model, inputs)

        elif acq_name == "ei":
            return EI(model, inputs)

        elif acq_name == "us":
            return US(model, inputs)

        elif acq_name == "us_bo":
            return US_BO(model, inputs)

        elif acq_name == "us_lw":
            return US_LW(model, inputs)

        elif acq_name == "us_iw":
            likelihood = Likelihood(model, inputs, weight_type="nominal")
            a = US_LW(model, inputs, likelihood=likelihood)
            return a
        
        elif acq_name == "us_my_acq2__0_0_1000":
            likelihood2 = Likelihood(model, inputs, weight_type="importance_ho", c_w2=0, c_w3=1000, tol=1e-2)
            a = US_LW(model, inputs, likelihood=likelihood2)
            return a

        elif acq_name == "us_lwbo":
            return US_LWBO(model, inputs)

        elif acq_name == "lcb":
            return LCB(model, inputs)

        elif acq_name == "lcb_lw":
            return LCB_LW(model, inputs)

        elif acq_name == "ivr":
            return IVR(model, inputs)

        elif acq_name == "ivr_bo":
            return IVR_BO(model, inputs)

        elif acq_name == "ivr_iw":
            likelihood = Likelihood(model, inputs, weight_type="nominal")
            # plt.figure()
            # plt.plot(likelihood.evaluate(inputs.draw_samples(n_samples=int(200), sample_method="grd")))
            a = IVR_LW(model, inputs, likelihood=likelihood)
            return a
        
        elif acq_name == "ivr_ho": # Alireza Mojahed
            likelihood = Likelihood(model, inputs, weight_type="importance_ho") # Alireza Mojahed
            a = IVR_LW(model, inputs, likelihood=likelihood) # Alireza Mojahed
            return a # Alireza Mojahed

        elif acq_name == "ivr_lw":
            return IVR_LW(model, inputs)
        
        elif acq_name == "my_acq1_0":
            return my_acq1_0(model, inputs)
        
        elif acq_name == "my_acq1__0_0_1":
            return my_acq1_0(model, inputs, c_w2=0, c_w3=1, tol=1e-5)
        
        elif acq_name == "my_acq1__0_0_10":
            return my_acq1_0(model, inputs, c_w2=0, c_w3=10, tol=1e-4)
        
        elif acq_name == "my_acq1__0_0_100":
            return my_acq1_0(model, inputs, c_w2=0, c_w3=100, tol=1e-3)
        
        elif acq_name == "my_acq1__0_0_1000":
            return my_acq1_0(model, inputs, c_w2=0, c_w3=1000, tol=1e-2)
        
        elif acq_name == "my_acq1__0_1_10":
            return my_acq1_0(model, inputs, c_w2=1, c_w3=10, tol=1e-4)
        
        elif acq_name == "my_acq1__0_1_100":
            return my_acq1_0(model, inputs, c_w2=1, c_w3=100, tol=1e-3)
        
        elif acq_name == "my_acq1__0_1_1000":
            return my_acq1_0(model, inputs, c_w2=1, c_w3=1000, tol=1e-2)
        
        elif acq_name == "my_acq1_1":
            return my_acq1_1(model, inputs)
        
        elif acq_name == "my_acq1__1_0_1":
            return my_acq1_1(model, inputs, c_w2=0, c_w3=1, tol=1e-5)
        
        elif acq_name == "my_acq1__1_0_10":
            return my_acq1_1(model, inputs, c_w2=0, c_w3=10, tol=1e-4)
        
        elif acq_name == "my_acq1__1_0_100":
            return my_acq1_1(model, inputs, c_w2=0, c_w3=100, tol=1e-3)
        
        elif acq_name == "my_acq1__1_0_1000":
            return my_acq1_1(model, inputs, c_w2=0, c_w3=1000, tol=1e-2)
        
        elif acq_name == "my_acq1__1_1_10":
            return my_acq1_1(model, inputs, c_w2=1, c_w3=10, tol=1e-4)
        
        elif acq_name == "my_acq1__1_1_100":
            return my_acq1_1(model, inputs, c_w2=1, c_w3=100, tol=1e-3)
        
        elif acq_name == "my_acq1__1_1_1000":
            return my_acq1_1(model, inputs, c_w2=1, c_w3=1000, tol=1e-2)
        
        elif acq_name == "my_acq2_0":
            return my_acq2_0(model, inputs)
        
        elif acq_name == "my_acq2__0_0_1":
            return my_acq2_0(model, inputs, c_w2=0, c_w3=1, tol=1e-5)
        
        elif acq_name == "my_acq2__0_0_10":
            return my_acq2_0(model, inputs, c_w2=0, c_w3=10, tol=1e-4)
        
        elif acq_name == "my_acq2__0_0_100":
            return my_acq2_0(model, inputs, c_w2=0, c_w3=100, tol=1e-3)
        
        elif acq_name == "my_acq2__0_0_1000":
            return my_acq2_0(model, inputs, c_w2=0, c_w3=1000, tol=1e-2)
        
        elif acq_name == "my_acq2__0_1_1":
            return my_acq2_0(model, inputs, c_w2=1, c_w3=1, tol=1e-5)
        
        elif acq_name == "my_acq2__0_1_10":
            return my_acq2_0(model, inputs, c_w2=1, c_w3=10, tol=1e-4)
        
        elif acq_name == "my_acq2__0_1_100":
            return my_acq2_0(model, inputs, c_w2=1, c_w3=100, tol=1e-3)
        
        elif acq_name == "my_acq2__0_1_1000":
            return my_acq2_0(model, inputs, c_w2=1, c_w3=1000, tol=1e-2)
        
        elif acq_name == "my_acq2_1":
            return my_acq2_1(model, inputs)
        
        elif acq_name == "my_acq2__1_0_1":
            return my_acq2_1(model, inputs, c_w2=0, c_w3=1, tol=1e-5)
        
        elif acq_name == "my_acq2__1_0_10":
            return my_acq2_1(model, inputs, c_w2=0, c_w3=10, tol=1e-4)
        
        elif acq_name == "my_acq2__1_0_100":
            return my_acq2_1(model, inputs, c_w2=0, c_w3=100, tol=1e-3)
        
        elif acq_name == "my_acq2__1_0_1000":
            return my_acq2_1(model, inputs, c_w2=0, c_w3=1000, tol=1e-2)
        
        elif acq_name == "my_acq2__1_1_10":
            return my_acq2_1(model, inputs, c_w2=1, c_w3=10, tol=1e-4)
        
        elif acq_name == "my_acq2__1_1_100":
            return my_acq2_1(model, inputs, c_w2=1, c_w3=100, tol=1e-3)
        
        elif acq_name == "my_acq2__1_1_1000":
            return my_acq2_1(model, inputs, c_w2=1, c_w3=1000, tol=1e-2)
        
        elif acq_name == "my_acq1_0_nt":
            return my_acq1_0(model, inputs, trig_id=False)
        
        elif acq_name == "my_acq1_1_nt":
            return my_acq1_1(model, inputs, trig_id=False)
        
        elif acq_name == "my_acq2_0_nt":
            return my_acq2_0(model, inputs, trig_id=False)
        
        elif acq_name == "my_acq2_1_nt":
            return my_acq2_1(model, inputs, trig_id=False)

        elif acq_name == "ivr_lwbo":
            return IVR_LWBO(model, inputs)

        elif acq_name == "us_lwraw":
            likelihood = Likelihood(model, inputs, fit_gmm=False)
            a = US_LW(model, inputs, likelihood=likelihood)
            return a

        elif acq_name == "us_lwboraw":
            likelihood = Likelihood(model, inputs, fit_gmm=False)
            a = US_LWBO(model, inputs, likelihood=likelihood)
            return a

        elif acq_name == "lcb_lwraw":
            likelihood = Likelihood(model, inputs, fit_gmm=False)
            a = LCB_LW(model, inputs, likelihood=likelihood)
            return a

        else:
            raise NotImplementedError

    elif isinstance(acquisition, Acquisition):
        return acquisition

    elif issubclass(acquisition, Acquisition):
        return acquisition(model, inputs)

    else:
        raise ValueError


