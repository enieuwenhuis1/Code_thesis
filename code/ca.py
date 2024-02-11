import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(y,t):

    k = 0.3

    for i in range(2, 100, 2):
        if t > i:
            k += 0.02


    print(k)
    dydt = -k * y
    return dydt

# initial condition
y0 = 20

# time points
t = np.linspace(0,20, 21)
print(t)
# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()
