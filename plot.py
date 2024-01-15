import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
# import cvxpy as cp
import numpy as np
import pandas as pd
from mpmath import diff
import scipy
import gekko

if __name__ == '__main__':
    biceps = [86.50, 93.64, 92.71, 100.08, 111.40, 86.50, 86.50, 95.00, 86.50, 195.59]
    brachialis = [333.38, 621.04, 494.26, 391.59, 431.90, 555.32, 494.04, 357.78, 498.69, 157.00]
    brachiorad = [450.00, 450.00, 450.00, 450.00, 449.44, 415.00, 450.00, 450.00, 379.04, 450.00]
    plt.plot(biceps, label='biceps')
    plt.plot(brachialis, label='brachialis')
    plt.plot(brachiorad, label='brachiorad')
    plt.legend()
    plt.ylabel('F_max(N)')
    plt.show()
