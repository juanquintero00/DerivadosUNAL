import numpy as np
import pandas as pd
from scipy.stats import norm

def bsm_prc(payoff_type, X0, K, r, sigma, T, q=0, side="buy"):
    d1 = (np.log(X0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if payoff_type == "call" or payoff_type == "c":
      price = X0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif payoff_type == "put" or payoff_type == "p":
      price = K * np.exp(-r * T) * norm.cdf(-d2) - X0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
      raise ValueError("payoff_type debe ser 'call'/'c' o 'put'/'p'")
    if side == "short" or side == "sell":
      price = price*-1
    return price

def GBM_paths(n_simul, T, t_steps, mu, sigma, X0, discret_method="Euler-Maruyama", plot=False, out_pd=False, **kwargs):
  """
  Genera n_simul trayectorias de un Movimiento Browniano Geometrico, de la forma
    dX(t) =  mu*X(t)*dt + sigma*X(t)*dWt
  
  :param mu (float): Drift del proceso (constante)
  :param sigma (float): Volatilidad del proceso (constante)
  :param T (float): Horizonte de tiempo
  :param t_steps (float): Numero de pasos en el horizonte de tiempo
  :param discret_method (string): Metodo de discretizacion del proceso, por ahora solo contempla
    'Euler-Maruyama' o 'Milstein', por defecto se usa 'Euler-Maruyama'
  :return (pd DataFrame): n_simul trayectorias del proceso usando el metodo de discretizacion seleccionado (en forma vertical)
  """
  dt = T/t_steps
  X = np.zeros((int(t_steps + 1), n_simul))
  X[0,:] = X0

  if discret_method == "Euler-Maruyama" or discret_method == "E-M":
    for t in range(1, X.shape[0]):
      dW = np.sqrt(dt) * np.random.normal(loc=0, scale=1, size=X.shape[1])
      X[t,:] = X[t-1,:] + mu*X[t-1,:]*dt + sigma*np.multiply(X[t-1,:],dW)
  elif discret_method == "Milstein":
    for t in range(1, X.shape[0]):
      dW = np.sqrt(dt) * np.random.normal(loc=0, scale=1, size=X.shape[1])
      X[t,:] = X[t-1,:] + mu*X[t-1,:]*dt + sigma*np.multiply(X[t-1,:],dW) + 0.5*(sigma**2)*np.multiply(X[t-1,:], (dW**2)-dt)
  else:
    raise ValueError("Metodo de discretizacion no implementado, solo es posible 'Euler-Maruyama'/'E-M' o 'Milstein'")

  if plot:
    pd.DataFrame(X, index=np.linspace(0, T, t_steps+1)).plot(legend=False)
  if out_pd:
    out = pd.DataFrame(X, index=np.linspace(0, T, t_steps+1))
    return out
  else:
    return X

def logGBM_paths(n_simul, T, t_steps, mu, sigma, X0, discret_method="Euler-Maruyama", plot=False, out_pd=False, **kwargs):
  """
  Genera n_simul trayectorias de un Movimiento Browniano Geometrico Lognormal, de la forma
    d lnX(t) =  (mu-(sigma^2/2))dt + sigma*dWt
  
  :param mu (float): Drift del proceso (constante)
  :param sigma (float): Volatilidad del proceso (constante)
  :param T (float): Horizonte de tiempo
  :param t_steps (float): Numero de pasos en el horizonte de tiempo
  :param discret_method (string): Metodo de discretizacion del proceso, por ahora solo contempla
    'Euler-Maruyama' o 'Milstein', por defecto se usa 'Euler-Maruyama'
  :return (pd DataFrame): n_simul trayectorias del proceso usando el metodo de discretizacion seleccionado (en forma vertical)
  """
  dt = T/t_steps
  X = np.zeros((int(t_steps + 1), n_simul))
  X[0,:] = X0

  if discret_method == "Euler-Maruyama" or discret_method == "E-M" or discret_method == "Milstein":
    for t in range(1, X.shape[0]):
      dW = np.sqrt(dt) * np.random.normal(loc=0, scale=1, size=X.shape[1])
      X[t,:] = np.multiply(X[t-1,:],np.exp((mu - 0.5*(sigma**2))*dt + sigma*dW))
  else:
    raise ValueError("Metodo de discretizacion no implementado, solo es posible 'Euler-Maruyama'/'E-M' o 'Milstein'")

  if plot:
    pd.DataFrame(X, index=np.linspace(0, T, t_steps+1)).plot(legend=False)
  if out_pd:
    out = pd.DataFrame(X, index=np.linspace(0, T, t_steps+1))
    return out
  else:
    return X

def ConsDrift_paths(n_simul, T, t_steps, mu, sigma, X0, discret_method="Euler-Maruyama", plot=False, out_pd=False, **kwargs):
  """
  Genera n_simul trayectorias de un Proceso con Drift Constante, de la forma
    dX(t) =  mu*dt + sigma*dWt
  
  :param mu (float): Drift del proceso (constante)
  :param sigma (float): Volatilidad del proceso (constante)
  :param T (float): Horizonte de tiempo
  :param t_steps (float): Numero de pasos en el horizonte de tiempo
  :param discret_method (string): Metodo de discretizacion del proceso, por ahora solo contempla
    'Euler-Maruyama' o 'Milstein', por defecto se usa 'Euler-Maruyama'
  :return (pd DataFrame): n_simul trayectorias del proceso usando el metodo de discretizacion seleccionado (en forma vertical)
  """
  dt = T/t_steps
  X = np.zeros((int(t_steps + 1), n_simul))
  X[0,:] = X0

  if discret_method == "Euler-Maruyama" or discret_method == "E-M" or discret_method == "Milstein":
    for t in range(1, X.shape[0]):
      dW = np.sqrt(dt) * np.random.normal(loc=0, scale=1, size=X.shape[1])
      X[t,:] = X[t-1,:] + mu*dt + sigma*dW
  else:
    raise ValueError("Metodo de discretizacion no implementado, solo es posible 'Euler-Maruyama'/'E-M' o 'Milstein'")

  if plot:
    pd.DataFrame(X, index=np.linspace(0, T, t_steps+1)).plot(legend=False)
  if out_pd:
    out = pd.DataFrame(X, index=np.linspace(0, T, t_steps+1))
    return out
  else:
    return X

def europ_call(side, paths_arr, K, r, T, **kwargs):
  last_prices = paths_arr[(paths_arr.shape[0]-1),:]
  n_simul = paths_arr.shape[1]
  if side == "buy" or side == "long":
    payoff = np.maximum(last_prices-np.repeat(K, n_simul),np.repeat(0, n_simul))
  elif side == "sell" or side == "short":
    payoff = -1*np.maximum(last_prices-np.repeat(K, n_simul),np.repeat(0, n_simul))
  else:
    raise ValueError("No esta bien especificado el lado de compra, debe ser 'buy'/'long' o 'sell'/'short'")
  mean = np.exp(-r*T)*np.mean(payoff)
  std = np.std(payoff)
 
  return mean, std, n_simul

def europ_put(side, paths_arr, K, r, T, **kwargs):
  last_prices = paths_arr[(paths_arr.shape[0]-1),:]
  n_simul = paths_arr.shape[1]
  if side == "buy" or side == "long":
    payoff = np.maximum(np.repeat(K, n_simul)-last_prices,np.repeat(0, n_simul))
  elif side == "sell" or side == "short":
    payoff = -1*np.maximum(np.repeat(K, n_simul)-last_prices,np.repeat(0, n_simul))
  else:
    raise ValueError("No esta bien especificado el lado de compra, debe ser 'buy'/'long' o 'sell'/'short'")
  mean = np.exp(-r*T)*np.mean(payoff)
  std = np.std(payoff)
 
  return mean, std, n_simul

def lookb_call(side, strike_type, strike_fun, paths_arr, r, T, last_prc_K="all", **kwargs):
  last_prices = paths_arr[(paths_arr.shape[0]-1),:]
  n_simul = paths_arr.shape[1]
  if (strike_fun == "min" and last_prc_K == "all"):
    strike_vec = np.min(paths_arr, axis=0)
  elif (strike_fun == "min" and last_prc_K != "all"):
    init = paths_arr.shape[0]-1-last_prc_K
    end = paths_arr.shape[0]-1
    strike_vec = np.max(paths_arr[init:end,:], axis=0)
  elif (strike_fun == "max" and last_prc_K == "all"):
    strike_vec = np.max(paths_arr, axis=0)
  elif (strike_fun == "max" and last_prc_K != "all"):
    init = paths_arr.shape[0]-1-last_prc_K
    end = paths_arr.shape[0]-1
    strike_vec = np.max(paths_arr[init:end,:], axis=0)
  else:
    raise ValueError("El argumento 'strike_fun' solo soporta 'min' o 'max' para payoffs tipo Lookback")
  if (side == "buy" or side == "long") and strike_type == "fix":
    try:
      K = kwargs.pop('K')
    except:
      raise ValueError("Si es strike es fijo (fix), debe incluir K dentro de los argumentos")
    payoff = np.maximum(strike_vec-np.repeat(K, n_simul),np.repeat(0, n_simul))
  elif (side == "buy" or side == "long") and strike_type == "float":
    payoff = np.maximum(last_prices-strike_vec,np.repeat(0, n_simul))
  elif (side == "sell" or side == "short") and strike_type == "fix":
    try:
      K = kwargs.pop('K')
    except:
      raise ValueError("Si es strike es fijo (fix), debe incluir K dentro de los argumentos")
    payoff = -1*np.maximum(strike_vec-np.repeat(K, n_simul),np.repeat(0, n_simul))
  elif (side == "sell" or side == "short") and strike_type == "float":
    payoff = -1*np.maximum(last_prices-strike_vec,np.repeat(0, n_simul))
  else:
    raise ValueError("No esta bien especificado el lado de compra, debe ser 'buy'/'long' o 'sell'/'short'. O 'strike_type', este debe ser 'fix' o 'float'")
  mean = np.exp(-r*T)*np.mean(payoff)
  std = np.std(payoff)
 
  return mean, std, n_simul

def lookb_put(side, strike_type, strike_fun, paths_arr, r, T, last_prc_K="all", **kwargs):
  last_prices = paths_arr[(paths_arr.shape[0]-1),:]
  n_simul = paths_arr.shape[1]
  if (strike_fun == "min" and last_prc_K == "all"):
    strike_vec = np.min(paths_arr, axis=0)
  elif (strike_fun == "min" and last_prc_K != "all"):
    init = paths_arr.shape[0]-1-last_prc_K
    end = paths_arr.shape[0]-1
    strike_vec = np.max(paths_arr[init:end,:], axis=0)
  elif (strike_fun == "max" and last_prc_K == "all"):
    strike_vec = np.max(paths_arr, axis=0)
  elif (strike_fun == "max" and last_prc_K != "all"):
    init = paths_arr.shape[0]-1-last_prc_K
    end = paths_arr.shape[0]-1
    strike_vec = np.max(paths_arr[init:end,:], axis=0)
  else:
    raise ValueError("El argumento 'strike_fun' solo soporta 'min' o 'max' para payoffs tipo Lookback")
  if (side == "buy" or side == "long") and strike_type == "fix":
    try:
      K = kwargs.pop('K')
    except:
      raise ValueError("Si es strike es fijo (fix), debe incluir K dentro de los argumentos")
    payoff = np.maximum(np.repeat(K, n_simul)-strike_vec,np.repeat(0, n_simul))
  elif (side == "buy" or side == "long") and strike_type == "float":
    payoff = np.maximum(strike_vec-last_prices,np.repeat(0, n_simul))
  elif (side == "sell" or side == "short") and strike_type == "fix":
    try:
      K = kwargs.pop('K')
    except:
      raise ValueError("Si es strike es fijo (fix), debe incluir K dentro de los argumentos")
    payoff = -1*np.maximum(np.repeat(K, n_simul)-strike_vec,np.repeat(0, n_simul))
  elif (side == "sell" or side == "short") and strike_type == "float":
    payoff = -1*np.maximum(strike_vec-last_prices,np.repeat(0, n_simul))
  else:
    raise ValueError("No esta bien especificado el lado de compra, debe ser 'buy'/'long' o 'sell'/'short'. O 'strike_type', este debe ser 'fix' o 'float'")
  mean = np.exp(-r*T)*np.mean(payoff)
  std = np.std(payoff)
 
  return mean, std, n_simul

def asian_call(side, strike_type, strike_fun, paths_arr, r, T, last_prc_K="all", **kwargs):
  last_prices = paths_arr[(paths_arr.shape[0]-1),:]
  n_simul = paths_arr.shape[1]
  if (strike_fun == "arit" and last_prc_K == "all"):
    strike_vec = np.mean(paths_arr, axis=0)
  elif (strike_fun == "arit" and last_prc_K != "all"):
    init = paths_arr.shape[0]-1-last_prc_K
    end = paths_arr.shape[0]-1
    strike_vec = np.max(np.mean(paths_arr[init:end,:], axis=0), axis=0)
  elif (strike_fun == "geom" and last_prc_K == "all"):
    strike_vec = np.exp(np.mean(np.log(paths_arr), axis=0))
  elif (strike_fun == "geom" and last_prc_K != "all"):
    init = paths_arr.shape[0]-1-last_prc_K
    end = paths_arr.shape[0]-1
    strike_vec = np.exp(np.mean(np.log(paths_arr[init:end,:]), axis=0))
  else:
    raise ValueError("El argumento 'strike_fun' solo soporta 'arit' o 'geom' para payoffs tipo Asiaticas")
  if (side == "buy" or side == "long") and strike_type == "fix":
    try:
      K = kwargs.pop('K')
    except:
      raise ValueError("Si es strike es fijo (fix), debe incluir K dentro de los argumentos")
    payoff = np.maximum(strike_vec-np.repeat(K, n_simul),np.repeat(0, n_simul))
  elif (side == "buy" or side == "long") and strike_type == "float":
    payoff = np.maximum(last_prices-strike_vec,np.repeat(0, n_simul))
  elif (side == "sell" or side == "short") and strike_type == "fix":
    try:
      K = kwargs.pop('K')
    except:
      raise ValueError("Si es strike es fijo (fix), debe incluir K dentro de los argumentos")
    payoff = -1*np.maximum(strike_vec-np.repeat(K, n_simul),np.repeat(0, n_simul))
  elif (side == "sell" or side == "short") and strike_type == "float":
    payoff = -1*np.maximum(last_prices-strike_vec,np.repeat(0, n_simul))
  else:
    raise ValueError("No esta bien especificado el lado de compra, debe ser 'buy'/'long' o 'sell'/'short'. O 'strike_type', este debe ser 'fix' o 'float'")
  mean = np.exp(-r*T)*np.mean(payoff)
  std = np.std(payoff)
 
  return mean, std, n_simul

def asian_put(side, strike_type, strike_fun, paths_arr, r, T, last_prc_K="all", **kwargs):
  last_prices = paths_arr[(paths_arr.shape[0]-1),:]
  n_simul = paths_arr.shape[1]
  if (strike_fun == "arit" and last_prc_K == "all"):
    strike_vec = np.mean(paths_arr, axis=0)
  elif (strike_fun == "arit" and last_prc_K != "all"):
    init = paths_arr.shape[0]-1-last_prc_K
    end = paths_arr.shape[0]-1
    strike_vec = np.max(np.mean(paths_arr[init:end,:], axis=0), axis=0)
  elif (strike_fun == "geom" and last_prc_K == "all"):
    strike_vec = np.exp(np.mean(np.log(paths_arr), axis=0))
  elif (strike_fun == "geom" and last_prc_K != "all"):
    init = paths_arr.shape[0]-1-last_prc_K
    end = paths_arr.shape[0]-1
    strike_vec = np.exp(np.mean(np.log(paths_arr[init:end,:]), axis=0))
  else:
    raise ValueError("El argumento 'strike_fun' solo soporta 'arit' o 'geom' para payoffs tipo Asiaticas")
  if (side == "buy" or side == "long") and strike_type == "fix":
    try:
      K = kwargs.pop('K')
    except:
      raise ValueError("Si es strike es fijo (fix), debe incluir K dentro de los argumentos")
    payoff = np.maximum(np.repeat(K, n_simul)-strike_vec,np.repeat(0, n_simul))
  elif (side == "buy" or side == "long") and strike_type == "float":
    payoff = np.maximum(strike_vec-last_prices,np.repeat(0, n_simul))
  elif (side == "sell" or side == "short") and strike_type == "fix":
    try:
      K = kwargs.pop('K')
    except:
      raise ValueError("Si es strike es fijo (fix), debe incluir K dentro de los argumentos")
    payoff = -1*np.maximum(np.repeat(K, n_simul)-strike_vec,np.repeat(0, n_simul))
  elif (side == "sell" or side == "short") and strike_type == "float":
    payoff = -1*np.maximum(strike_vec-last_prices,np.repeat(0, n_simul))
  else:
    raise ValueError("No esta bien especificado el lado de compra, debe ser 'buy'/'long' o 'sell'/'short'. O 'strike_type', este debe ser 'fix' o 'float'")
  mean = np.exp(-r*T)*np.mean(payoff)
  std = np.std(payoff)
 
  return mean, std, n_simul

def binary(side, strike_side, payment, paths_arr, r, T, K, last_prc_K="all", **kwargs):
  last_prices = paths_arr[(paths_arr.shape[0]-1),:]
  n_simul = paths_arr.shape[1]
  if payment == "cash":
    try:
      Q = kwargs.pop('Q')
    except:
      raise ValueError("Si es una opcion cash-or-nothing, se debe ingresar el cash que gana (Q)")
    if (strike_side == "call" or strike_side == "c") and (side == "buy" or side == "long"):
      payoff = Q*(last_prices >= K)
    elif (strike_side == "call" or strike_side == "c") and (side == "sell" or side == "short"):
      payoff = -Q*(last_prices >= K)
    if (strike_side == "put" or strike_side == "p") and (side == "buy" or side == "long"):
      payoff = Q*(K >= last_prices)
    elif (strike_side == "put" or strike_side == "p") and (side == "sell" or side == "short"):
      payoff = -Q*(K >= last_prices)
  elif payment == "asset":
    if (strike_side == "call" or strike_side == "c") and (side == "buy" or side == "long"):
      payoff = np.multiply((last_prices >= K), last_prices)
    elif (strike_side == "call" or strike_side == "c") and (side == "sell" or side == "short"):
      payoff = -np.multiply((last_prices >= K), last_prices)
    if (strike_side == "put" or strike_side == "p") and (side == "buy" or side == "long"):
      payoff = np.multiply((K >= last_prices), last_prices)
    elif (strike_side == "put" or strike_side == "p") and (side == "sell" or side == "short"):
      payoff = -np.multiply((K >= last_prices), last_prices)
  mean = np.exp(-r*T)*np.mean(payoff)
  std = np.std(payoff)
 
  return mean, std, n_simul

def barrier(side, strike_side, knock_type, paths_arr, r, T, K, H, last_prc_K="all", **kwargs):
  last_prices = paths_arr[(paths_arr.shape[0]-1),:]
  n_simul = paths_arr.shape[1]
  if knock_type == "down-and-out":
    eval = np.min(paths_arr, axis=0) >= H
  elif knock_type == "down-and-in":
    eval = np.min(paths_arr, axis=0) < H
  elif knock_type == "up-and-out":
    eval = np.max(paths_arr, axis=0) <= H
  elif knock_type == "up-and-in":
    eval = np.max(paths_arr, axis=0) > H
  else:
    raise ValueError("knock_type debe ser 'down-and-out', 'down-and-in', 'up-and-out' o 'up-and-in'")
  if (side == "long" and strike_side == "call") or (side == "buy" and strike_side == "call"):
    payoff = eval*np.multiply(eval,np.maximum(last_prices-np.repeat(K, n_simul), np.repeat(0, n_simul)))
  if (side == "short" and strike_side == "call") or (side == "sell" and strike_side == "call"):
    payoff = -1*eval*np.multiply(eval,np.maximum(last_prices-np.repeat(K, n_simul), np.repeat(0, n_simul)))
  if (side == "long" and strike_side == "put") or (side == "buy" and strike_side == "put"):
    payoff = eval*np.multiply(eval,np.maximum(np.repeat(K, n_simul)-last_prices, np.repeat(0, n_simul)))
  if (side == "short" and strike_side == "put") or (side == "sell" and strike_side == "put"):
    payoff = -1*eval*np.multiply(eval,np.maximum(np.repeat(K, n_simul)-last_prices, np.repeat(0, n_simul)))
  mean = np.exp(-r*T)*np.mean(payoff)
  std = np.std(payoff)
 
  return mean, std, n_simul

def MC_option_prc(side, payoff_fun, paths_fun, out_all=False, **kwargs):
  paths = paths_fun(**kwargs)
  price, std, N = payoff_fun(side=side, paths_arr=paths, **kwargs)
  if out_all:
    return price, std, N
  else:
    return price