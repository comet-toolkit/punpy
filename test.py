import punpy
import time
import numpy as np

# your measurement function
def calibrate_slow(L0,gains,dark):
      y2=np.repeat((L0-dark)*gains,1000)
      y2=y2+np.random.random(len(y2))
      y2=y2.sort()
      return (L0-dark)*gains

# your data
L0 = np.array([[[0.43,0.80,0.70,0.65,0.90],
               [0.41,0.82,0.73,0.64,0.93],
               [0.45,0.79,0.71,0.66,0.98]],
               [[0.42,0.83,0.69,0.64,0.88],
               [0.47,0.75,0.70,0.65,0.78],
               [0.45,0.86,0.72,0.66,0.86]],
               [[0.40,0.87,0.67,0.66,0.94],
               [0.39,0.80,0.70,0.65,0.87],
               [0.42,0.78,0.69,0.65,0.93]]])
dark = np.random.rand(3,3,5)*0.05
gains = np.tile(np.array([23,26,28,29,31]),(3,3,1)) # same gains as before, but repeated 10 times so that shapes match

# your uncertainties
L0_ur = np.array([[[0.02, 0.04, 0.02, 0.01, 0.06],
                  [0.02, 0.04, 0.02, 0.01, 0.06],
                  [0.02, 0.04, 0.02, 0.01, 0.06]],
                  [[0.02, 0.04, 0.02, 0.01, 0.06],
                  [0.02, 0.04, 0.02, 0.01, 0.06],
                  [0.02, 0.04, 0.02, 0.01, 0.06]],
                  [[0.02, 0.04, 0.02, 0.01, 0.06],
                  [0.02, 0.04, 0.02, 0.01, 0.06],
                  [0.02, 0.04, 0.02, 0.01, 0.06]]])

gains_ur = 0.02*gains  # 2% random uncertainty
gains_us = 0.03*gains  # 3% systematic uncertainty
dark_ur = np.ones((3,3,5))*0.02  # random uncertainty of 0.02

prop=punpy.LPUPropagation(parallel_cores=1)
L1=calibrate_slow(L0,gains,dark)
t1=time.time()

L1_ur=prop.propagate_random(calibrate_slow,[L0,gains,dark],
      [L0_ur,gains_ur,dark_ur],repeat_dims=[0,1])
t2=time.time()

print(L1)
print(L1_ur)
print("propogate_random took: ",t2-t1," s")