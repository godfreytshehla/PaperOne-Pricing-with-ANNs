#@Author: Godfrey Tshehla


#Import modules
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
        
### The BSM model class for vanilla European option
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
        

### Monte Carlo method class for vanilla European option
class MonteCarloEuropeanOption:
    default_steps = 500
    default_simulations = 100000

    def __init__(self, maturity: float, stockPrice: float, strikePrice: float, riskFreeRate: float, impliedVolatility: float, option_type: str):
        self.maturity = maturity
        self.stockPrice = stockPrice
        self.strikePrice = strikePrice
        self.riskFreeRate = riskFreeRate
        self.impliedVolatility = impliedVolatility
        self.option_type = option_type.lower()
        self.steps = MonteCarloEuropeanOption.default_steps
        self.simulations = MonteCarloEuropeanOption.default_simulations

    def simulate_stock_price(self):
        dt = self.maturity / self.steps
        half_simulations = self.simulations // 2
        price_paths = np.zeros((self.steps, self.simulations))
        price_paths[0, :] = self.stockPrice
        for t in range(1, self.steps):
            Z = np.random.standard_normal(half_simulations)
            antithetic_Z = -Z
            Z = np.concatenate((Z, antithetic_Z))  # Combine normal and antithetic variates
            price_paths[t] = price_paths[t - 1] * np.exp((self.riskFreeRate - 0.5 * self.impliedVolatility**2) * dt + self.impliedVolatility * np.sqrt(dt) * Z)
        return price_paths

    def price(self):
        terminal_prices = self.simulate_stock_price()[-1]
        payoffs = np.maximum(terminal_prices - self.strikePrice, 0) if self.option_type == 'call' else np.maximum(self.strikePrice - terminal_prices, 0)
        discounted_payoff = np.exp(-self.riskFreeRate * self.maturity) * np.mean(payoffs)
        return discounted_payoff

    def delta(self):
        dt = self.maturity / self.steps
        price_paths = self.simulate_stock_price()
        terminal_prices = price_paths[-1]
        if self.option_type == 'call':
            in_the_money = terminal_prices > self.strikePrice
            payoffs = np.exp(-self.riskFreeRate * self.maturity) * in_the_money * (price_paths[-1] / self.stockPrice)
        else:
            in_the_money = terminal_prices < self.strikePrice
            payoffs = -np.exp(-self.riskFreeRate * self.maturity) * in_the_money * (price_paths[-1] / self.stockPrice)
        return np.mean(payoffs)
        

        
### Binomial tree model for European option
class BinomialEuropeanOption:
    def __init__(self, maturity: float, stockPrice: float, strikePrice: float, riskFreeRate: float, impliedVolatility: float, Nperiod: int, option_type: str):
        self.maturity = maturity
        self.stockPrice = stockPrice
        self.strikePrice = strikePrice
        self.riskFreeRate = riskFreeRate
        self.impliedVolatility = impliedVolatility
        self.Nperiod = Nperiod
        self.option_type = option_type.lower()
        self.dt = self.maturity / self.Nperiod  # time step
        self.u = np.exp(self.impliedVolatility * np.sqrt(self.dt))  # up factor
        self.d = 1 / self.u  # down factor
        self.q = (np.exp(self.riskFreeRate * self.dt) - self.d) / (self.u - self.d)  # risk-neutral probability

    def __create_price_tree(self):
        price_tree = np.zeros([self.Nperiod + 1, self.Nperiod + 1])
        for i in range(self.Nperiod + 1):
            for j in range(i + 1):
                price_tree[j, i] = self.stockPrice * (self.u ** (i - j)) * (self.d ** j)
        return price_tree

    def __create_option_tree(self, price_tree):
        if self.option_type == 'call':
            option_tree = np.maximum(price_tree - self.strikePrice, 0)
        else:  # put option
            option_tree = np.maximum(self.strikePrice - price_tree, 0)
        
        for i in range(self.Nperiod - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = np.exp(-self.riskFreeRate * self.dt) * (self.q * option_tree[j, i + 1] + (1 - self.q) * option_tree[j + 1, i + 1])
        return option_tree

    def price(self):
        price_tree = self.__create_price_tree()
        option_tree = self.__create_option_tree(price_tree)
        return option_tree[0, 0]

    def delta(self):
        price_tree = self.__create_price_tree()
        option_tree = self.__create_option_tree(price_tree)
        delta = (option_tree[0, 1] - option_tree[1, 1]) / (price_tree[0, 1] - price_tree[1, 1])
        return delta

    
############################################################# Valuation of Chooser options #################################################

### BSM model class for Chooser option
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
        

### Monte Carlo method class for Chooser option
class MonteCarloChooserOption:
    def __init__(self, choiceTime: float, maturity: float, 
                 stockPrice: float, strikePrice: float, riskFreeRate: float, 
                 impliedVolatility: float, simulations: int = 100000, steps: int = 500):
        self.choiceTime = choiceTime
        self.maturity = maturity
        self.stockPrice = stockPrice
        self.strikePrice = strikePrice
        self.riskFreeRate = riskFreeRate
        self.impliedVolatility = impliedVolatility
        self.simulations = simulations
        self.steps = steps

    def simulate_stock_price(self):
        #dt = self.choiceTime / self.steps
        dt = self.maturity / self.steps
        price_paths = np.zeros((self.steps, self.simulations))
        price_paths[0, :] = self.stockPrice

        for t in range(1, self.steps):
            Z = np.random.standard_normal(self.simulations)
            price_paths[t] = price_paths[t - 1] * np.exp((self.riskFreeRate - 0.5 * self.impliedVolatility**2) * dt + self.impliedVolatility * np.sqrt(dt) * Z)
        
        return price_paths[-1]

    def calculate_payoff(self):
        final_prices = self.simulate_stock_price()
        # Decision for call or put at the choice time
        call_payoff = np.maximum(final_prices - self.strikePrice, 0)
        put_payoff = np.maximum(self.strikePrice - final_prices, 0)
        # Choose the higher payoff for each path
        payoffs = np.maximum(call_payoff, put_payoff)
        return payoffs

    def price(self):
        payoffs = self.calculate_payoff()
        # Discount the average payoff back to the present value
        discounted_payoff = np.exp(-self.riskFreeRate * self.maturity) * np.mean(payoffs)
        return discounted_payoff
    
    def delta(self):
        final_prices = self.simulate_stock_price()
        dt = self.maturity / self.steps

        # Compute the delta for each path
        deltas = np.zeros(self.simulations)
        for i in range(self.simulations):
            if final_prices[i] > self.strikePrice:
                # Path ends as a call option
                deltas[i] = np.exp(-self.riskFreeRate * self.maturity) * (final_prices[i] > self.strikePrice) * (final_prices[i] / self.stockPrice)
            else:
                # Path ends as a put option
                deltas[i] = -np.exp(-self.riskFreeRate * self.maturity) * (final_prices[i] < self.strikePrice) * (final_prices[i] / self.stockPrice)
                    # Calculate the average delta
        return np.mean(deltas) 

    
### Binomial tree model for Chooser option
class BinomialChooserOption:
    def __init__(self, choiceTime: float, maturity: float, stockPrice: float, strikePrice: float, riskFreeRate: float, impliedVolatility: float, Nperiod: int):
        self.maturity = maturity 
        self.choiceTime = choiceTime  
        self.stockPrice = stockPrice  
        self.strikePrice = strikePrice  
        self.riskFreeRate = riskFreeRate 
        self.impliedVolatility = impliedVolatility 
        self.Nperiod = Nperiod  
        self.dt = self.maturity / self.Nperiod 
        self.u = np.exp(self.impliedVolatility * np.sqrt(self.dt)) 
        self.d = 1 / self.u  
        self.q = (np.exp(self.riskFreeRate * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral probability

    def __create_price_tree(self):
        price_tree = np.zeros([self.Nperiod + 1, self.Nperiod + 1])
        for i in range(self.Nperiod + 1):
            for j in range(i + 1):
                price_tree[j, i] = self.stockPrice * (self.u ** (i - j)) * (self.d ** j)
        return price_tree

    def __create_option_trees(self, price_tree):
        call_tree = np.zeros([self.Nperiod + 1, self.Nperiod + 1])
        put_tree = np.zeros([self.Nperiod + 1, self.Nperiod + 1])

        for i in range(self.Nperiod + 1):
            call_tree[i, self.Nperiod] = max(price_tree[i, self.Nperiod] - self.strikePrice, 0)
            put_tree[i, self.Nperiod] = max(self.strikePrice - price_tree[i, self.Nperiod], 0)

        choice_point = int(self.choiceTime * self.Nperiod / self.maturity)
        for i in range(self.Nperiod - 1, -1, -1):
            for j in range(i + 1):
                call_tree[j, i] = np.exp(-self.riskFreeRate * self.dt) * (self.q * call_tree[j, i + 1] + (1 - self.q) * call_tree[j + 1, i + 1])
                put_tree[j, i] = np.exp(-self.riskFreeRate * self.dt) * (self.q * put_tree[j, i + 1] + (1 - self.q) * put_tree[j + 1, i + 1])
                if i == choice_point:
                    call_tree[j, i] = max(call_tree[j, i], put_tree[j, i])

        return call_tree

    def price(self):
        price_tree = self.__create_price_tree()
        call_tree = self.__create_option_trees(price_tree)
        return call_tree[0, 0]

    def delta(self):
        price_tree = self.__create_price_tree()
        call_tree = self.__create_option_trees(price_tree)
        delta = (call_tree[0, 1] - call_tree[1, 1]) / (price_tree[0, 1] - price_tree[1, 1])
        return delta
        
################################################### Valuation of Cash-or-nothing options ###################################################
        
### BSM model class for cash-or-nothing option
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
        

### Monte Carlo method class for cash-or-nothing option
class MonteCarloCashOrNothingOption:
    default_steps = 500
    default_simulations = 100000

    def __init__(self, maturity: float, stockPrice: float, strikePrice: float, riskFreeRate: float, impliedVolatility: float, payout: float, option_type: str):
        self.maturity = maturity
        self.stockPrice = stockPrice
        self.strikePrice = strikePrice
        self.riskFreeRate = riskFreeRate
        self.impliedVolatility = impliedVolatility
        self.payout = payout
        self.option_type = option_type.lower()
        self.steps = MonteCarloCashOrNothingOption.default_steps
        self.simulations = MonteCarloCashOrNothingOption.default_simulations

    def simulate_stock_price(self):
        dt = self.maturity / self.steps
        half_simulations = self.simulations // 2
        price_paths = np.zeros((self.steps, self.simulations))
        price_paths[0, :] = self.stockPrice
        for t in range(1, self.steps):
            Z = np.random.standard_normal(half_simulations)
            antithetic_Z = -Z
            Z = np.concatenate((Z, antithetic_Z))  # Combine normal and antithetic variates
            price_paths[t] = price_paths[t - 1] * np.exp((self.riskFreeRate - 0.5 * self.impliedVolatility**2) * dt + self.impliedVolatility * np.sqrt(dt) * Z)
        return price_paths

    def price(self):
        terminal_prices = self.simulate_stock_price()[-1]
        if self.option_type == 'call':
            payoffs = np.where(terminal_prices > self.strikePrice, self.payout, 0)
        else:
            payoffs = np.where(terminal_prices < self.strikePrice, self.payout, 0)
        discounted_payoff = np.exp(-self.riskFreeRate * self.maturity) * np.mean(payoffs)
        return discounted_payoff
    
    
    def delta(self):
        epsilon = 0.01  # Small change in the initial stock price

        # Bump the stock price up and down
        bumped_up_price = self.stockPrice * (1 + epsilon)
        bumped_down_price = self.stockPrice * (1 - epsilon)

        # Save original stock price and set to bumped up price
        original_stock_price = self.stockPrice
        self.stockPrice = bumped_up_price
        terminal_prices_up = self.simulate_stock_price()[-1]

        # Reset to original and set to bumped down price
        self.stockPrice = bumped_down_price
        terminal_prices_down = self.simulate_stock_price()[-1]

        # Reset to original price
        self.stockPrice = original_stock_price

        if self.option_type == 'call':
            payoffs_up = np.where(terminal_prices_up > self.strikePrice, self.payout, 0)
            payoffs_down = np.where(terminal_prices_down > self.strikePrice, self.payout, 0)
        else:  # put option
            payoffs_up = np.where(terminal_prices_up < self.strikePrice, self.payout, 0)
            payoffs_down = np.where(terminal_prices_down < self.strikePrice, self.payout, 0)

        delta_estimate = (np.mean(payoffs_up) - np.mean(payoffs_down)) / (2 * epsilon * self.stockPrice)
        return np.exp(-self.riskFreeRate * self.maturity) * delta_estimate
    
    
    
### Binomial tree model for Cash or nothing option
class BinomialCashOrNothingOption:
    def __init__(self, maturity: float, stockPrice: float, strikePrice: float, riskFreeRate: float, impliedVolatility: float, Nperiod: int, option_type: str):
        self.maturity = maturity
        self.stockPrice = stockPrice 
        self.strikePrice = strikePrice 
        self.riskFreeRate = riskFreeRate 
        self.impliedVolatility = impliedVolatility  
        self.Nperiod = Nperiod  
        self.option_type = option_type.lower()  
        self.dt = self.maturity / self.Nperiod  
        self.u = np.exp(self.impliedVolatility * np.sqrt(self.dt)) 
        self.d = 1 / self.u 
        self.q = (np.exp(self.riskFreeRate * self.dt) - self.d) / (self.u - self.d)  

    def __create_price_tree(self):
        price_tree = np.zeros([self.Nperiod + 1, self.Nperiod + 1])
        for i in range(self.Nperiod + 1):
            for j in range(i + 1):
                price_tree[j, i] = self.stockPrice * (self.u ** (i - j)) * (self.d ** j)
        return price_tree

    def __create_option_tree(self, price_tree):
        option_tree = np.zeros([self.Nperiod + 1, self.Nperiod + 1])
        if self.option_type == 'call':
            option_tree[:, self.Nperiod] = np.where(price_tree[:, self.Nperiod] > self.strikePrice, 1, 0)
        else:  # Put option
            option_tree[:, self.Nperiod] = np.where(price_tree[:, self.Nperiod] < self.strikePrice, 1, 0)

        for i in range(self.Nperiod - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = np.exp(-self.riskFreeRate * self.dt) * (self.q * option_tree[j, i + 1] + (1 - self.q) * option_tree[j + 1, i + 1])
        return option_tree

    def price(self):
        price_tree = self.__create_price_tree()
        option_tree = self.__create_option_tree(price_tree)
        return option_tree[0, 0]

    def delta(self):
        # Delta calculation for cash-or-nothing options is not straightforward
        # and should be interpreted with caution
        price_tree = self.__create_price_tree()
        option_tree = self.__create_option_tree(price_tree)
        delta = (option_tree[0, 1] - option_tree[1, 1]) / (price_tree[0, 1] - price_tree[1, 1])
        return delta
    

############################################################# Synthetic data generation ####################################################    
    
### Data generation class
class GenerateSyntheticData:
    
    def __init__(self, nsamples: int):
        self.nsamples = nsamples

    def data(self):
        # Generating random data
        stockPrice = np.random.uniform(50, 500, self.nsamples)  # Spot price
        strikePrice = np.random.uniform(40, 700, self.nsamples)  # Strike price
        maturity = np.random.uniform(1/252, 5, self.nsamples)  # Time to maturity in years
        choiceTime = np.random.uniform(1/252, maturity, self.nsamples) # generate time to choose for the chooser option
        riskFreeRate = np.random.uniform(0.01, 0.2, self.nsamples)  # Risk-free interest rate
        impliedVolatility = np.random.uniform(0.1, 0.8, self.nsamples)  # Volatility

        # Combining the data into a single array for scaling
        data = np.vstack((maturity, stockPrice, strikePrice, riskFreeRate, impliedVolatility, choiceTime))

        return data