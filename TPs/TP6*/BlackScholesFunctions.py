# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps

def Put_BS_core(tau, K, DF, F, sigma):
    """
    Black-Scholes put option price in terms of 
    
    tau   : time to maturity
    K     : strike
    DF    : discount factor. DF = e^{-r*tau} if constant interest rate r
    F     : underlying forward price
    sigma : volatility
    """
    sigma_sqrt_tau = sigma * np.sqrt(tau)
    
    d_1 = np.log(F/K)/sigma_sqrt_tau + sigma_sqrt_tau/2.
    
    d_2 = d_1 - sigma_sqrt_tau

    prix_put = DF * (K * sps.norm.cdf(-d_2) - F * sps.norm.cdf(-d_1))

    return prix_put


def Call_BS_core(tau, K, DF, F, sigma):
    """
    Black-Scholes call option price in terms of 
    
    tau   : time to maturity
    K     : strike
    DF    : discount factor. DF = e^{-r*tau} if constant interest rate r
    F     : underlying forward price
    sigma : volatility
    """
    sigma_sqrt_tau = sigma * np.sqrt(tau)

    d_1 = np.log(F/K)/sigma_sqrt_tau + sigma_sqrt_tau/2.
    
    d_2 = d_1 - sigma_sqrt_tau

    prix_call = DF * (F * sps.norm.cdf(d_1) - K * sps.norm.cdf(d_2))

    return prix_call


def Put_BS(t, S, T, K, r, sigma):
    """
    Black-Scholes put option price in terms of the standard BS parameters.
    """
    tau = T - t
    DF = np.exp(-r*tau)
    F = S * np.exp(r*tau)
    
    return Put_BS_core(tau, K, DF, F, sigma)


def Vega_core(tau, K, DF, F, sigma):
    """
    Vega of a Black-Scholes put (or call) in terms of
    
    tau   : time to maturity
    K     : strike
    DF    : discount factor
    F     : underlying forward price
    sigma : volatility
    """
    sigma_sqrt_tau = sigma * np.sqrt(tau)
    
    d_1 = np.log(F/K) / sigma_sqrt_tau + sigma_sqrt_tau / 2.

    vega = DF * F * np.sqrt(tau) * np.exp(-d_1**2 / 2) / np.sqrt(2*np.pi)
    
    return vega


def Vega(t, S, T, K, r, sigma):
    """
    Vega of a Black-Scholes put (or call) in terms of the standard BS parameters
    """
    tau = T - t
    DF = np.exp(-r*tau)
    F = S * np.exp(r*tau)
    
    return Vega_core(tau, K, DF, F, sigma)


def volImplCore_Newton(tau, K, DF, F, price,
                          CallOrPutFlag = 1,
                          initial_point=0.25, price_tol = 1.e-4, max_iter=50):
    """
    Implied volatility of a put or call option with price = price, the other parameters being
    
    tau   : time to maturity
    K     : strike
    DF    : discount factor
    F     : underlying forward price
    
    Method: Newton.
    """
    vol = initial_point
    
    if CallOrPutFlag:
        current_price = Call_BS_core(tau, K, DF, F, vol)
    else:
        current_price = Put_BS_core(tau, K, DF, F, vol)
    
    stopping_rule = np.abs(current_price - price)
    iterations = 0
    
    while ( (stopping_rule > price_tol) & (iterations < max_iter) ):
        iterations = iterations + 1
        
        vol = vol - (current_price - price) / Vega_core(tau, K, DF, F, vol)
        
        if CallOrPutFlag:
            current_price = Call_BS_core(tau, K, DF, F, vol)
        else:
            current_price = Put_BS_core(tau, K, DF, F, vol)
        
        stopping_rule = np.abs(current_price - price)
    
    return vol, iterations


def volImplPutCore_bisection(tau, K, DF, F, price, price_tol = 1.e-3, max_iter=50, a = 0.001, b = 2.):
    """
    Implied volatility of a put option with price = price, the other parameters being
    
    tau   : time to maturity
    K     : strike
    DF    : discount factor
    F     : underlying forward price
    
    Method: bisection over the interval [a, b]. 
    """
    prix_min = Put_BS_core(tau, K, DF, F, a)
    prix_max = Put_BS_core(tau, K, DF, F, b)
    
    check = (prix_min < price < prix_max)
    
    if check == False:
        print("""WARNING: Option price outside the given interval [a,b].
            The implied volatility evaluation is impossible for the given bounds.
            This function is returning vol = 0.""")
        return 0
    
    else:
        vol_min = a
        vol_max = b
        
        vol = (vol_min + vol_max) / 2
        mid_interval_price = Put_BS_core(tau, K, DF, F, vol)
        
        stopping_rule = np.abs(mid_interval_price - price)
        iterations = 0
        
        while ( (stopping_rule > price_tol) & (iterations < max_iter) ):
            iterations = iterations + 1
            
            if mid_interval_price - price > 0:
                vol_max = vol
                vol = (vol_min + vol_max) / 2
            else:
                vol_min = vol
                vol = (vol_min + vol_max) / 2
            
            mid_interval_price = Put_BS_core(tau, K, DF, F, vol)
            
            stopping_rule = np.abs(mid_interval_price - price)
        
        return vol, iterations


def volImplCallCore_bisection(tau, K, DF, F, price, price_tol = 1.e-3, max_iter=50, a = 0.001, b = 2.):
    """
    Implied volatility of a call option with price = price, the other parameters being
    
    tau   : time to maturity
    K     : strike
    DF    : discount factor
    F     : underlying forward price
    
    Method: bisection over the interval [a, b]. 
    """
    prix_min = Call_BS_core(tau, K, DF, F, a)
    prix_max = Call_BS_core(tau, K, DF, F, b)
    
    check = (prix_min < price < prix_max)
    
    if check == False:
        print("""WARNING: Option price outside the given interval [a,b].
            The implied volatility evaluation is impossible for the given bounds.
            This function is returning vol = 0.""")
        return 0
    
    else:
        vol_min = a
        vol_max = b
        
        vol = (vol_min + vol_max) / 2
        mid_interval_price = Call_BS_core(tau, K, DF, F, vol)
        
        stopping_rule = np.abs(mid_interval_price - price)
        iterations = 0
        
        while ( (stopping_rule > price_tol) & (iterations < max_iter) ):
            iterations = iterations + 1
            
            if mid_interval_price - price > 0:
                vol_max = vol
                vol = (vol_min + vol_max) / 2
            else:
                vol_min = vol
                vol = (vol_min + vol_max) / 2
            
            mid_interval_price = Call_BS_core(tau, K, DF, F, vol)
            
            stopping_rule = np.abs(mid_interval_price - price)
        
        return vol, iterations