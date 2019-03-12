from typing import NamedTuple, Union
import numpy as np

class Option(NamedTuple):
    call: bool # true if call option, false if put
    american: bool # true if american option, false if european
    S: float # underlying asset price
    K: float # strike price
    sigma: float # standard deviation of the return for the stock price
    tau: float # time to maturity (T-t), in years
    r: float # annual interest rate
    q: float #dividend yield


def monte_carlo_stock(option: Option, m: int, n: int) -> np.ndarray:
    # takes in an option and creates m Monte-Carlo price path simulations of the underlying, for n time-steps
    # mu is the drift of the underlying

    # initialize the stock price matrix and set the first value to be the value of the underlying
    S = np.zeros((m, n + 1))
    S[:, 0] = option.S
    # delta_t is the length of the time steps, i.e. the time to maturity divided by the umber of steps
    delta_t = option.tau / n

    # simulate a matrix of returns
    returns = np.random.normal((option.r - np.square(option.sigma) / 2.) * delta_t, option.sigma * np.sqrt(delta_t),
                               size=(m, n))

    for j in range(1, n + 1):
        # start at 1 since we have S_0, and then simulate n time-steps
        S[:, j] = np.multiply(S[:, j - 1], np.exp(returns[:, j - 1]))

    return S


def payoff(s: Union[float, np.ndarray], option: Option) -> Union[float, np.ndarray]:
    # returns the payoff for an american option given price of underlying is s
    if option.call:
        return np.maximum(s - option.K, 0)

    return np.maximum(option.K - s, 0)


def Binary_Tree(option: Option, steps: int) -> float:
    # function that recursively finds the price of an American option
    assert (steps >= 0)

    delta_t = option.tau / steps
    up = np.exp(option.sigma * np.sqrt(delta_t))
    down = 1 / up
    # probability of upwards movement
    p = (np.exp((option.r - option.q) * delta_t) - down) / (up - down)

    # calculate the discount rate per time step
    gamma = np.exp(-option.r * delta_t)

    return binary_tree_helper(gamma, p, option.S, up, down, option, steps)


def binary_tree_helper(gamma: float, p: float, s: float, up: float, down: float, option: Option, steps: int) -> float:
    # helper function to calculate the option price
    # feed in the price
    if steps <= 0:
        # we have reached the end
        if option.call:
            return np.maximum(s - option.K, 0)
        return np.maximum(option.K - s, 0)

    # recursively find the price of the next step
    price_up = binary_tree_helper(gamma, p, s * up, up, down, option, steps - 1)
    price_down = binary_tree_helper(gamma, p, s * down, up, down, option, steps - 1)

    if option.american:
        # check if option is american
        if option.call:
            # check if call option
            return np.maximum(gamma * (p * price_up + (1 - p) * price_down), s - option.K)
        return np.maximum(gamma * (p * price_up + (1 - p) * price_down), option.K - s)

    # if it reaches here it is a european option
    return gamma * (p * price_up + (1 - p) * price_down)


def longstaff_schwartz(option: Option, m: int, n: int) -> float:
    # uses the longstaff-schwartz algorithm to return the value of an american option

    delta_t = option.tau / n
    disc = np.exp(-option.r * delta_t)
    # initialize the value function
    cf = np.zeros((m, 1))
    # simulate the stock price
    SP = monte_carlo_stock(option, m, n)

    # set the initial value of the value function to the payoff at maturity
    cf = payoff(SP[:, -1], option)

    # recursively backtrack the value of the option
    for j in range(n - 1, 0, -1):
        cf = cf * disc
        # only add feature functions for paths that are in the money
        indices = np.array([i for i in range(m) if payoff(SP[i, j], option) > 0]).reshape(-1, 1)
        X = feature_func(SP[indices, j], option)
        Y = cf[indices].reshape(-1, 1)

        if np.size(X) > 0:
            # regress the non-linear features of stock price and strike price onto the value function
            estimate = X.dot(np.linalg.lstsq(X, Y, rcond=None)[0])
            pay = payoff(SP[indices, j], option).reshape(-1, 1)
            # set the value function to be equal to payoff if the payoff is higher than the estimated future value
            # of holding onto the option, otherwise let it remain as the discounted value function
            cf[indices] = np.where(pay > estimate, pay, cf[indices])

    # calculate the exercise vs continuation values
    exercise = payoff(option.S, option)
    cont = np.mean(cf * disc)

    return np.maximum(exercise, cont)


def feature_func(s: Union[float, np.ndarray], option: Option) -> np.ndarray:
    # takes in price of underlying and an option and returns a feature function
    sp = np.divide(s, option.K).reshape(-1, 1)
    # uses the feature functions suggested by Longstaff-Schwartz
    phi_0 = np.ones((np.size(sp), 1))
    phi_1 = np.exp(-sp / 2)
    phi_2 = np.multiply(np.exp(-sp / 2), 1 - sp)
    phi_3 = np.multiply(np.exp(-sp / 2), 1 - 2 * sp + np.square(sp) / 2)

    return np.concatenate((phi_0, phi_1, phi_2, phi_3), axis=1)