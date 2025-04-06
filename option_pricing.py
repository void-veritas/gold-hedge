import numpy as np
from scipy.stats import norm


def calculate_put_option_price(spot_price, strike_price, time_to_expiry, volatility, risk_free_rate=0.02):
    """
    Calculate the price of a put option using the Black-Scholes model.

    Parameters:
    -----------
    spot_price : float
        Current price of the underlying asset
    strike_price : float
        Strike price of the option
    time_to_expiry : float
        Time to expiration in years
    volatility : float
        Annualized volatility of the underlying asset
    risk_free_rate : float, optional
        Risk-free interest rate, default is 0.02 (2%)

    Returns:
    --------
    float
        The estimated put option price
    """
    # Calculate d1 and d2 parameters for Black-Scholes
    d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (
                volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    # Calculate put price using Black-Scholes formula
    put_price = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)

    return put_price