import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import scipy
from scipy import signal
from scipy.optimize import minimize
from pyomo.environ import *
from pyomo.opt import SolverFactory

# from emg_distribution import *
# from test import *
# from require import *

length = 500
folder = 'files/squat/output/trials/'
plot_distribution = True


if __name__ == '__main__':
    mat = read_all_mat('training')
    dis = emg_distribution()
    opt = optimization_pyomo_squat(mat['MmtArm'], mat['torque'], mat['activation'], mat['iso'], mat['fv'])
    plot_all_result_squat(int(mat['torque'].shape[0] / length), opt['torque'], opt['activation'], mat['torque'],
                          mat['activation'], mat['time'], dis['mean'], dis['std'])
    before_motion(opt['fmax'], dis['mean'], dis['std'])
    application(opt['fmax'], dis['mean'], dis['std'])

    plt.show()
