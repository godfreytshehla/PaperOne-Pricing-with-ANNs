#@Author: Godfrey Tshehla


#Py modules used
import numpy as np
from scipy.stats import norm


############################################## Valuation of the Zero-Coupon Bond ##########################################################

class ZeroCouponBond:
    def __init__(self, maturity: float, riskFreeRate: float):
        self.riskFreeRate = riskFreeRate
        self.maturity = maturity

    def price(self):
        return np.exp(-self.riskFreeRate * self.maturity)
               
############################################ Valuation of the vanilla European option ######################################################
        
class AnalyticalVanillaEuropeanOption:
    def __init__(self, maturity: float, stockPrice: float, strikePrice: float, riskFreeRate: float, impliedVolatility: float, option_type: str):
        self.maturity = maturity
        self.stockPrice = stockPrice
        self.strikePrice = strikePrice
        self.riskFreeRate = riskFreeRate
        self.impliedVolatility = impliedVolatility
        self.option_type = option_type.lower()

    def __compute_d1_d2(self):
        d1 = (np.log(self.stockPrice / self.strikePrice) + 
              (self.riskFreeRate + 0.5 * self.impliedVolatility**2) * 
              self.maturity) / (self.impliedVolatility * np.sqrt(self.maturity))
        d2 = d1 - self.impliedVolatility * np.sqrt(self.maturity)
        return d1, d2
    
    def price(self):
        d1, d2 = self.__compute_d1_d2()
        if self.option_type == 'call':
            price = (self.stockPrice * norm.cdf(d1) - 
                     self.strikePrice * np.exp(-self.riskFreeRate * self.maturity) * 
                     norm.cdf(d2))
        else:  # Put option
            price = (self.strikePrice * np.exp(-self.riskFreeRate * self.maturity) * 
                     norm.cdf(-d2) - self.stockPrice * norm.cdf(-d1))
        return price
    
    def delta(self):
        d1, _ = self.__compute_d1_d2()
        if self.option_type == 'call':
            delta = norm.cdf(d1)
        else:  # Put option
            delta = -norm.cdf(-d1)
        return delta
    
############################################################# Valuation of Chooser options #################################################

class AnalyticalChooserOption:
    def __init__(self, choiceTime: float, maturity: float, 
                 stockPrice: float, strikePrice: float, riskFreeRate: float, impliedVolatility: float):
        self.choiceTime = choiceTime
        self.maturity = maturity
        self.stockPrice = stockPrice
        self.strikePrice = strikePrice
        self.riskFreeRate = riskFreeRate
        self.impliedVolatility = impliedVolatility

    def __d_plus(self):
        T = self.maturity
        return ((np.log(self.stockPrice / self.strikePrice) + 
                (self.riskFreeRate + (self.impliedVolatility**2) / 2) * T) / 
                (self.impliedVolatility * np.sqrt(T)))

    def __d_minus(self, d_plus):
        T = self.maturity
        return d_plus - self.impliedVolatility * np.sqrt(T)

    def __d1(self):
        t = self.choiceTime
        T = self.maturity
        return ((np.log(self.stockPrice / self.strikePrice) + 
                 self.riskFreeRate * T + (self.impliedVolatility**2 / 2) * t) / 
                (self.impliedVolatility * np.sqrt(t)))

    def __d2(self, d1):
        return d1 - self.impliedVolatility * np.sqrt(self.choiceTime)

    def price(self):
        d_plus = self.__d_plus()
        d_minus = self.__d_minus(d_plus)
        d1 = self.__d1()
        d2 = self.__d2(d1)

        value = (self.stockPrice * (norm.cdf(d_plus) - norm.cdf(-d1)) +
                 self.strikePrice * np.exp(-self.riskFreeRate * self.maturity) * 
                 (norm.cdf(-d2) - norm.cdf(d_minus)))
        return value
    
    def delta(self):
        d_plus = self.__d_plus()
        d1 = self.__d1()
        return norm.cdf(d_plus) - norm.cdf(-d1)
        
################################################### Valuation of Cash-or-nothing options ###################################################

class AnalyticalCashOrNothingOption:
    def __init__(self, maturity: float, stockPrice: float, strikePrice: float, riskFreeRate: float, impliedVolatility: float, payout: float, option_type: str):
        self.maturity = maturity
        self.stockPrice = stockPrice
        self.strikePrice = strikePrice
        self.riskFreeRate = riskFreeRate
        self.impliedVolatility = impliedVolatility
        self.payout = payout
        self.kappa = 1 if option_type.lower() == 'call' else -1

    def __d_minus(self):
        return ((np.log(self.stockPrice / self.strikePrice) + 
                (self.riskFreeRate - (self.impliedVolatility**2) / 2) * self.maturity) / 
                (self.impliedVolatility * np.sqrt(self.maturity)))

    def price(self):
        d_minus = self.__d_minus()
        value = self.payout * np.exp(-self.riskFreeRate * self.maturity) * norm.cdf(self.kappa * d_minus)
        return value
    
    def delta(self):
        d_minus = self.__d_minus()
        phi_d_minus = norm.pdf(d_minus)  # Standard normal probability density function at d_minus
        delta = self.kappa * self.payout * np.exp(-self.riskFreeRate * self.maturity) * \
                phi_d_minus / (self.impliedVolatility * self.stockPrice * np.sqrt(self.maturity))
        return delta
    
    
############################################################# Synthetic data generation ####################################################    
class GenerateSyntheticData:
    
    def __init__(self, nsamples: int):
        self.nsamples = nsamples
        self.seed = 42  # Default seed value set for reproducibility
        np.random.seed(self.seed)

    def data(self):
        # Generating random data
        stockPrice = np.random.uniform(50, 500, self.nsamples)  # Spot price
        strikePrice = np.random.uniform(40, 700, self.nsamples)  # Strike price
        maturity = np.random.uniform(1/252, 5, self.nsamples)  # Time to maturity in years
        choiceTime = np.random.uniform(1/252, maturity, self.nsamples) # Time to choose for the chooser option
        riskFreeRate = np.random.uniform(0.01, 0.2, self.nsamples)  # Risk-free interest rate
        impliedVolatility = np.random.uniform(0.1, 0.8, self.nsamples)  # Volatility

        # Combining the data into a single array for scaling (we use min-max scaling)
        data = np.vstack((maturity, stockPrice, strikePrice, riskFreeRate, impliedVolatility, choiceTime))

        return data