import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
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
i = 10
r = 0


def cost(parameters, cases, recovered, S0, I0, R0):
    def SIR(t, y):
        S, I, R = y
        dS = -beta * S * I
        dR = gamma * I
        dI = dS - dR
        return dS, dI, dR

    beta, gamma = parameters
    res = solve_ivp(SIR, [0, len(cases)], (S0, I0, R0), t_eval=np.arange(0, len(cases), 1), vectorized=True)

    mse_cases = mean_squared_error(res.y[0], cases)
    mse_recovered = mean_squared_error(res.y[1], recovered)
    return mse_cases + mse_recovered


b, g = minimize(cost, [0.25, 1/21], args=(cases, recovered, s, i, r), bounds=[(0.01, 0.5), (1/21, 1/7)],
                options={"maxiter": 1})

