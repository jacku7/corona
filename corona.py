import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, odeint
from sklearn.metrics import mean_squared_error


cases = pd.read_csv("cases.csv")
recovered = pd.read_csv("recovered.csv")


def stuff(data):
    data = data.query('Country == "Canada"').copy()
    data.loc['Total', :] = data.sum(axis=0)
    return data.loc['Total'][4:]


cases = stuff(cases)
recovered = stuff(recovered)

s = 37674666
# s = 330617954
e = 10
i = 0
r = 0
N = s + e + i + r


def cost(parameters, S0, E0, I0, R0):
    def deriv(y, t):
        # S, E, I, R = y
        # dSdt = -beta * S * I / N
        # dIdt = beta * S * I / N - gamma * I
        # dRdt = gamma * I
        # return dSdt, dIdt, dRdt
        S, E, I, R = y
        dSdt = - beta * (S * I) / N
        dEdt = beta * (S * I) / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    beta, gamma, sigma = parameters
    y0 = S0, E0, I0, R0
    t = np.linspace(0, len(cases), len(cases))
    ret = odeint(deriv, y0, t)

    S, E, I, R = ret.T
    mse_cases = mean_squared_error(I, cases)
    mse_recovery = mean_squared_error(R, recovered)

    return mse_cases + mse_recovery

test = minimize(cost, (0.12908239, 0.0266665 , 1.82139425), args=(s, e, i, r), bounds=[(0, 1), (0, 1), (0, 10)])

print(test)


