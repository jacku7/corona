import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 37058856
# Initial number of infected and recovered individuals, I0 and R0.
I0 = 1
R0 = 0
# initial exposed
E0 = 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - E0
# 1 / incubation period
alpha = 1. / 14
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta = 0.5
gamma = 1. / 7
# A grid of time points (in days)
t = np.linspace(0, 320, 320)

# The SIR model differential equations.
def deriv(y, t, N, alpha, beta, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, E0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, alpha, beta, gamma))
S, E, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', label='Susceptible')
ax.plot(t, E/N, 'y', label='Exposed')
ax.plot(t, I/N, 'r', label='Infected')
ax.plot(t, R/N, 'g', label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('percentage of population')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_ylim(0,1)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()