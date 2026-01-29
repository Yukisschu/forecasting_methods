
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Running_average
# =============================

def running_average(y):
    """Compute the running (cumulative) average of a sequence."""
    running_avg = []                    # List to store running averages
    cumulative_sum = 0.0                # Accumulator for the sum

    for i, v in enumerate(y, start=1):  # Loop over values with 1-based index
        cumulative_sum += v             # Add current value to cumulative sum
        running_avg.append(cumulative_sum / i)  # Append average up to i

    return running_avg                  # Return running average list


def running_average_forecast(y):
    """One-step-ahead running-average forecast using data up to t-1."""
    ra = np.empty_like(y, dtype=float)  # Allocate array for forecasts
    ra[0] = np.nan                      # Forecast undefined at first index
    csum = 0.0                          # Cumulative sum

    for i in range(len(y)):             # Loop over all observations
        csum += y[i]                    # Update cumulative sum
        if i >= 1:                      # Forecast defined from second observation
            ra[i] = (csum - y[i]) / i   # Average of values up to t-1

    return ra                           # Return forecast array

# =============================
# Random walk
# =============================
def lag_forecast(y):
    """Random walk (lag-1) forecast: Ŷ_t = Y_{t-1}."""
    n = len(y)                     # Length of series
    yhat = [None] * n                   # Initialize forecast list

    for t in range(1, n):               # Loop from second observation
        yhat[t] = float(y[t - 1])  # Forecast equals previous value

    return yhat                         # Return forecasts



# =============================
# Exponential smoothing
# =============================
def exp_smoothing_forecast(y, alpha):
    """One-step-ahead simple exponential smoothing forecast."""
    es = np.empty_like(y, dtype=float)   # Allocate forecast array
    es[0] = y[0]                          # Initialize level

    for t in range(1, len(y)):           # Loop over time
        es[t] = alpha * y[t - 1] + (1.0 - alpha) * es[t - 1]  # Update rule

    es[0] = np.nan                        # Forecast undefined at first index
    return es                             # Return forecast series


# def estimate_alpha_exponential_smoothing(y, criterion="MSE", grid_size=2000, eps=1e-4):
#     """
#     Expanding-window exponential smoothing with re-estimated alpha at each step.

#     For each t (starting at 2), use data y[:t] to grid-search alpha that minimizes the
#     chosen criterion over one-step-ahead forecasts within that subsample, then produce
#     a single one-step-ahead forecast for time t (i.e., forecast of y[t] using y[:t]).

#     Returns
#     -------
#     alpha_t : np.ndarray
#         Chosen alpha for each time index (NaN where not defined).
#     yhat_t : np.ndarray
#         One-step-ahead forecast series (NaN where not defined).
#     u_t : np.ndarray
#         One-step-ahead forecast errors y - yhat (NaN where not defined).
#     """
#     y = np.asarray(y, dtype=float)
#     T = len(y)

#     alphas = np.linspace(eps, 1.0, grid_size)

#     alpha_t = np.full(T, np.nan)
#     yhat_t = np.full(T, np.nan)
#     u_t = np.full(T, np.nan)

#     # Need at least 2 points to define one-step-ahead errors in this convention
#     for t in range(2, T):
#         y_sub = y[:t]  

#         best_alpha = None
#         best_val = np.inf

#         for a in alphas:
#             yhat_sub = exp_smoothing_forecast(y_sub, a)
#             u_sub = forecast_errors(y_sub, yhat_sub)
#             m = forecast_metrics(u_sub, y_sub, tau=2)

#             if criterion == "ME":
#                 val = m["ME"]
#             elif criterion == "MAE":
#                 val = m["MAE"]
#             elif criterion == "MAPE":
#                 val = m["MAPE"]
#             elif criterion == "MSE":
#                 val = m["MSE"]
#             else:
#                 raise ValueError("criterion must be one of: 'ME', 'MAE', 'MAPE', 'MSE'")

#             if val < best_val:
#                 best_val = val
#                 best_alpha = a

#         # Forecast y[t] using the best alpha estimated from y[:t]
#         yhat_sub_best = exp_smoothing_forecast(y_sub, best_alpha)

#         yhat_next = best_alpha * y_sub[-1] + \
#             (1 - best_alpha) * yhat_sub_best[-1]
        
#         print(
#             f"t={t}, best alpha={best_alpha:.4f}, "
#             f"y_sub={y_sub}, "
#             f"yhat_sub_best={yhat_sub_best}"
#         )

#         yhat_t[t] = yhat_next          # one-step-ahead forecast for index t
#         alpha_t[t] = best_alpha                # store chosen alpha
#         u_t[t] = y[t] - yhat_t[t]              # realized one-step-ahead error

#     return alpha_t, yhat_t, u_t



# def estimate_alpha_exponential_smoothing(y, criterion="MSE", grid_size=2000, eps=1e-4):
#     """
#     Expanding-window exponential smoothing with re-estimated alpha at each step.

#     For each t (starting at 2), use data y[:t] to grid-search alpha that minimizes the
#     chosen criterion over one-step-ahead forecasts within that subsample, then produce
#     a single one-step-ahead forecast for time t (i.e., forecast of y[t] using y[:t]).

#     Returns
#     -------
#     alpha_t : np.ndarray
#         Chosen alpha for each time index (NaN where not defined).
#     yhat_t : np.ndarray
#         One-step-ahead forecast series (NaN where not defined).
#     u_t : np.ndarray
#         One-step-ahead forecast errors y - yhat (NaN where not defined).
#     """
#     y = np.asarray(y, dtype=float)
#     T = len(y)

#     alphas = np.linspace(eps, 1.0, grid_size)

#     alpha_t = np.full(T, np.nan)
#     yhat_t = np.full(T, np.nan)
#     u_t = np.full(T, np.nan)

#     # Need at least 2 points to define one-step-ahead errors in this convention
#     for t in range(2, T):
#         y_sub = y[:t+1]  

#         best_alpha = None
#         best_val = np.inf

#         for a in alphas:
#             yhat_sub = exp_smoothing_forecast(y_sub, a)
#             u_sub = forecast_errors(y_sub, yhat_sub)
#             m = forecast_metrics(u_sub, y_sub, tau=2)

#             if criterion == "ME":
#                 val = m["ME"]
#             elif criterion == "MAE":
#                 val = m["MAE"]
#             elif criterion == "MAPE":
#                 val = m["MAPE"]
#             elif criterion == "MSE":
#                 val = m["MSE"]
#             else:
#                 raise ValueError("criterion must be one of: 'ME', 'MAE', 'MAPE', 'MSE'")

#             if val < best_val:
#                 best_val = val
#                 best_alpha = a

#         # Forecast y[t] using the best alpha estimated from y[:t]
#         yhat_sub_best = exp_smoothing_forecast(y_sub, best_alpha)
        
#         # print(
#         #     f"t={t}, best alpha={best_alpha:.4f}, "
#         #     f"y_sub={y_sub}, "
#         #     f"yhat_sub_best={yhat_sub_best}"
#         # )

#         yhat_t[t] = yhat_sub_best[-1]          # one-step-ahead forecast for index t
#         alpha_t[t] = best_alpha                # store chosen alpha
#         u_t[t] = y[t] - yhat_t[t]              # realized one-step-ahead error

#     return alpha_t, yhat_t, u_t


def estimate_alpha_exponential_smoothing(y, criterion="MSE", grid_size=2000, eps=1e-4):
    y = np.asarray(y, dtype=float)
    T = len(y)

    alphas = np.linspace(eps, 1.0, grid_size)

    alpha_t = np.full(T, np.nan)  # store alpha used to forecast y[t]
    yhat_t  = np.full(T, np.nan)  # yhat_t[t] forecasts y[t]
    u_t     = np.full(T, np.nan)  # u_t[t] = y[t] - yhat_t[t]

    # At time t, estimate alpha from y[:t+1], then forecast y[t+1]
    for t in range(2, T - 1):   
        y_sub = y[:t+1]         # data available up to time t

        best_alpha = None
        best_val = np.inf

        for a in alphas:
            yhat_sub = exp_smoothing_forecast(y_sub, a)
            u_sub = forecast_errors(y_sub, yhat_sub)
            m = forecast_metrics(u_sub, y_sub, tau=2)

            if criterion == "ME":
                val = m["ME"]
            elif criterion == "MAE":
                val = m["MAE"]
            elif criterion == "MAPE":
                val = m["MAPE"]
            elif criterion == "MSE":
                val = m["MSE"]
            else:
                raise ValueError("criterion must be one of: 'ME', 'MAE', 'MAPE', 'MSE'")

            if val < best_val:
                best_val = val
                best_alpha = a

        # Now forecast y[t+1] using alpha estimated from y[:t+1]
        y_sub_t = y[:t+2]  # exclude y[t+2] 
        yhat_sub_best = exp_smoothing_forecast(y_sub_t, best_alpha)

        # print(
        #     f"# ={t+1:3d} | "
        #     f"len(y_sub)={len(y_sub):3d} | "
        #     f"len(yhat_sub)={len(yhat_sub_best):3d} | "
        # )

        yhat_t[t+1] = yhat_sub_best[-1]     # forecast for index t+1
        alpha_t[t+1] = best_alpha
        u_t[t+1] = y[t+1] - yhat_t[t+1]

    return alpha_t, yhat_t, u_t


# =============================
# Trend / regression-based
# =============================
def ar1_expanding_ols_forecast(y):
    """AR(1) expanding-window OLS one-step-ahead forecast."""
    ar1 = np.empty_like(y, dtype=float)   # Allocate forecast array
    ar1[:] = np.nan                       # Initialize with NaNs

    for t in range(2, len(y)):            # Start when enough data exists
        y_dep = y[1:t]                    # Dependent variable
        x_lag = y[0:t-1]                  # Lagged regressor
        X = np.column_stack([np.ones_like(x_lag), x_lag])  # Design matrix with intercept
        beta, _, _, _ = np.linalg.lstsq(X, y_dep, rcond=None)  # OLS
        c_hat, phi_hat = beta             # Extract coefficients
        ar1[t] = c_hat + phi_hat * y[t - 1]  # Forecast

    return ar1                            # Return AR(1) forecasts


def running_trend_forecast(y):
    """Running trend expanding-window OLS forecast."""
    y = np.asarray(y, dtype=float)        # Convert to array
    T = len(y)                            # Series length

    a_hat = np.full(T, np.nan)            # Intercept estimates
    b_hat = np.full(T, np.nan)            # Slope estimates
    forecast = np.full(T, np.nan)         # Forecast array

    for idx in range(2, T):               # Require at least two observations
        n = idx                           # Number of observations used
        j = np.arange(1, n + 1)           # Time index
        y_past = y[:n]                    # Past observations

        j_bar = j.mean()                  # Mean of time index 
        y_bar = y_past.mean()             # Mean of data

        denom = np.sum((j - j_bar) ** 2)  # Denominator for slope
        if denom == 0:
            continue                      # Skip if undefined

        b = np.sum((j - j_bar) * y_past) / denom  # Slope estimate
        a = y_bar - b * j_bar             # Intercept estimate

        a_hat[idx] = a                    # Store intercept
        b_hat[idx] = b                    # Store slope

        t_next = idx + 1                  # Next time index
        forecast[idx] = a + b * t_next    # Forecast

    return forecast, a_hat, b_hat         # Return outputs


def random_walk_plus_drift_forecast(y):
    """Random walk and random walk plus drift expanding-window forecasts."""
    y = np.asarray(y, dtype=float)      # Convert input to NumPy array
    T = len(y)                           # Length of time series

    rw = np.full(T, np.nan)              # Allocate random walk forecast
    rw[1:] = y[:-1]                      # Lagged values as forecasts

    dY = np.diff(y)                      # First differences of the series
    mu_hat = np.full(T, np.nan)          # Allocate drift estimates

    for t in range(2, T):                # Loop from third observation
        mu_hat[t] = dY[:t-1].mean()      # Mean of past differences

    rw_drift = np.full(T, np.nan)        # Allocate RW+drift forecast
    for t in range(2, T):                # Loop over valid indices
        rw_drift[t] = y[t-1] + mu_hat[t] # Add drift to lagged value

    return rw, rw_drift, mu_hat          # Return forecasts and drift


# =============================
# Holt-Winters (double exponential smoothing)
# =============================

def holt_forecast_from_series(y, alpha, beta):
    """
    Return Holt one-step-ahead forecast after full sample. 
    Returns one single number: the next-step forecast after the end of the sample.
    """
    y = np.asarray(y, dtype=float)        # Convert input to array
    if len(y) < 2:
        raise ValueError("Need at least two observations")

    L = y[0]                              # Initial level
    G = y[1] - y[0]                       # Initial trend

    for yt in y[1:]:                      # Loop through data
        L_prev = L                        # Store previous level
        L = alpha * yt + (1 - alpha) * (L + G)  # Update level
        G = beta * (L - L_prev) + (1 - beta) * G  # Update trend

    return L + G                          # Return forecast

def holt_observation_weights(alpha, beta, max_lag=40, burn=200):
    """
    lag 0 corresponds to Y_{t-1} (forecast formed using data up to t-1),
    not Y_t.
    """
    # choose a time index far from initialization
    t = burn + max_lag + 1

    # series ends at (t-1), length = t
    T = t

    weights = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        y = np.zeros(T)
        y[(t - 1) - k] = 1.0  # impulse at Y_{t-1-k}
        weights[k] = holt_forecast_from_series(y, alpha, beta)  # forecast for time t
    
    return weights


def holt_fitted_forecast_series(y, alpha, beta):
    """
    Return Holt fitted one-step-ahead forecast series. 
    Produces a forecast at every time step.
    """
    y = np.asarray(y, dtype=float)        # Convert to array

    L = y[0]                              # Initial level
    G = y[1] - y[0]                       # Initial trend

    F = np.full(len(y), np.nan)           # Allocate forecast array
    F[0] = y[0]                           # Initialize first value

    for t in range(1, len(y)):            # Loop over time
        F[t] = L + G                      # One-step-ahead forecast
        L_prev = L                        # Store previous level
        L = alpha * y[t] + (1 - alpha) * (L + G)  # Update level
        G = beta * (L - L_prev) + (1 - beta) * G  # Update trend

    return F                              # Return fitted forecasts




def estimate_alpha_beta_holt_winters(y, criterion="SSE", grid_n=201, eps=1e-3):
    y = np.asarray(y, dtype=float)
    T = len(y)

    grid = np.linspace(eps, 1.0 - eps, grid_n)

    alpha_t = np.full(T, np.nan)
    beta_t = np.full(T, np.nan)
    yhat_t = np.full(T, np.nan)
    u_t = np.full(T, np.nan)
    loss_t = np.full(T, np.nan)

    # Need enough data to initialize Holt inside holt_fitted_forecast_series (uses y[0] and y[1])
    # and to produce a one-step-ahead forecast for index t (so t must be at least 2).
    for t in range(2, T - 1):
        y_sub = y[:t+2]  

        best_alpha = None
        best_beta = None
        best_loss = np.inf

        for alpha in grid:
            for beta in grid:
                F = holt_fitted_forecast_series(y_sub, alpha, beta)

                y_eval = y_sub[1:]
                F_eval = F[1:]
                u = y_eval - F_eval

                if criterion == "ME":
                    loss = float(np.mean(u))
                elif criterion == "MAE":
                    loss = float(np.mean(np.abs(u)))
                elif criterion == "MAPE":
                    loss = float(np.mean(100.0 * np.abs(u) / np.abs(y_eval)))
                elif criterion == "MSE":
                    loss = float(np.mean(u ** 2))
                elif criterion == "SSE":
                    loss = float(np.sum(u ** 2))
                else:
                    raise ValueError("criterion must be one of: 'ME', 'MAE', 'MAPE', 'MSE', 'SSE'")

                if loss < best_loss:
                    best_alpha = alpha
                    best_beta = beta
                    best_loss = loss

        # One-step-ahead forecast for index t using best (alpha, beta) fitted on y[:t]
        F_best = holt_fitted_forecast_series(y_sub, best_alpha, best_beta)
        yhat_t[t+1] = F_best[-1]
        u_t[t+1] = y[t] - yhat_t[t]

        alpha_t[t+1] = best_alpha
        beta_t[t+1] = best_beta
        loss_t[t+1] = best_loss

    return alpha_t, beta_t, yhat_t, u_t, loss_t


# def estimate_alpha_beta_holt_winters(y, criterion="SSE", grid_n=201, eps=1e-3):
#     """
#     Expanding-window Holt (double exponential smoothing) with re-estimated alpha/beta at each step.

#     Returns
#     -------
#     alpha_t : np.ndarray
#         Estimated alpha for each time index (NaN where not defined).
#     beta_t : np.ndarray
#         Estimated beta for each time index (NaN where not defined).
#     yhat_t : np.ndarray
#         One-step-ahead forecast series (NaN where not defined).
#     u_t : np.ndarray
#         One-step-ahead forecast errors y - yhat (NaN where not defined).
#     loss_t : np.ndarray
#         Best loss value achieved at each time index (NaN where not defined).
#     """
#     y = np.asarray(y, dtype=float)
#     T = len(y)

#     grid = np.linspace(eps, 1.0 - eps, grid_n)

#     alpha_t = np.full(T, np.nan)
#     beta_t = np.full(T, np.nan)
#     yhat_t = np.full(T, np.nan)
#     u_t = np.full(T, np.nan)
#     loss_t = np.full(T, np.nan)

#     # Need enough data to initialize Holt inside holt_fitted_forecast_series (uses y[0] and y[1])
#     # and to produce a one-step-ahead forecast for index t (so t must be at least 2).
#     for t in range(2, T):
#         y_sub = y[:t+1]  

#         best_alpha = None
#         best_beta = None
#         best_loss = np.inf

#         for alpha in grid:
#             for beta in grid:
#                 F = holt_fitted_forecast_series(y_sub, alpha, beta)

#                 y_eval = y_sub[1:]
#                 F_eval = F[1:]
#                 u = y_eval - F_eval

#                 if criterion == "ME":
#                     loss = float(np.mean(u))
#                 elif criterion == "MAE":
#                     loss = float(np.mean(np.abs(u)))
#                 elif criterion == "MAPE":
#                     loss = float(np.mean(100.0 * np.abs(u) / np.abs(y_eval)))
#                 elif criterion == "MSE":
#                     loss = float(np.mean(u ** 2))
#                 elif criterion == "SSE":
#                     loss = float(np.sum(u ** 2))
#                 else:
#                     raise ValueError("criterion must be one of: 'ME', 'MAE', 'MAPE', 'MSE', 'SSE'")

#                 if loss < best_loss:
#                     best_alpha = alpha
#                     best_beta = beta
#                     best_loss = loss

#         # One-step-ahead forecast for index t using best (alpha, beta) fitted on y[:t]
#         F_best = holt_fitted_forecast_series(y_sub, best_alpha, best_beta)
#         yhat_t[t] = F_best[-1]
#         u_t[t] = y[t] - yhat_t[t]

#         alpha_t[t] = best_alpha
#         beta_t[t] = best_beta
#         loss_t[t] = best_loss

#     return alpha_t, beta_t, yhat_t, u_t, loss_t



# =============================
# Seasonal time series
# =============================
# ============================================================
# 1) Seasonal Random Walk (SRW):  Ŷ_t = Y_{t-S}
# ============================================================

def seasonal_random_walk_forecast(y, S):
    """
    Seasonal random walk one-step-ahead forecasts.

    For t >= S, forecasts: yhat[t] = y[t-S].
    """
    y = np.asarray(y, dtype=float)
    T = len(y)

    yhat = np.full(T, np.nan)
    for t in range(S, T):
        yhat[t] = y[t - S]

    return yhat


# ============================================================
# 2) Seasonal Random Walk with Drift:
#    Ŷ_t = c_{t-1} + Y_{t-S}
#    c_t = (1/(t-1)) * sum_{j=0}^{t-S-1} (Y_{t-j} - Y_{t-j-S})
# ============================================================

def seasonal_random_walk_with_drift_forecast(y, S):
    """
    Seasonal random walk with drift (expanding-window) one-step-ahead forecasts.

    Returns
    -------
    yhat : np.ndarray
        One-step-ahead forecasts (NaN where undefined).
    c_hat : np.ndarray
        Drift estimates c_t (NaN where undefined).
    """
    y = np.asarray(y, dtype=float)
    T = len(y)

    yhat = np.full(T, np.nan)
    c_hat = np.full(T, np.nan)

    # For drift we need at least one seasonal difference, so start at t = S+1 (0-based)
    for t in range(S + 1, T):
        # estimate c_{t-1} using seasonal differences up to time t-1
        diffs = y[S:t] - y[:t - S]          # (Y_S - Y_0), ..., (Y_{t-1} - Y_{t-1-S})
        c_tm1 = diffs.mean()                # c_{t-1}
        c_hat[t] = c_tm1                    # store (aligned with forecast time index)
        yhat[t] = c_tm1 + y[t - S]          # forecast y[t]

    return yhat, c_hat


# ============================================================
# 3) Running Seasonal Regression:
#    Y_t = mu + beta t + sum_{j=1}^S gamma_j D_{jt} + eps_t
#    with sum gamma_j = 0  (use S-1 dummies, last is baseline)
# ============================================================

# def running_seasonal_regression_forecast(y, S):
#     """
#     Expanding-window OLS with linear trend and seasonal dummies (S-1 to avoid collinearity).

#     Uses t = 1..T time index and season = (t-1) mod S.
#     One-step-ahead forecast at time t uses data up to t-1.
#     """
#     y = np.asarray(y, dtype=float)
#     T = len(y)

#     yhat = np.full(T, np.nan)
#     a_hat = np.full(T, np.nan)   # intercept mu estimate
#     b_hat = np.full(T, np.nan)   # slope beta estimate
#     g_hat = np.full((T, S), np.nan)  # seasonal effects (full S; last reconstructed)

#     # Need at least S+1 points to estimate trend + seasonal dummies in a stable way
#     for t in range(S + 1, T):
#         y_past = y[:t]                           # y[0..t-1]
#         tpast = np.arange(1, t + 1, dtype=float) # 1..t

#         # Build seasonal dummy matrix for S-1 seasons (last season is baseline)
#         season_idx = (np.arange(t) % S)          # 0..S-1
#         D = np.zeros((t, S - 1), dtype=float)
#         for j in range(S - 1):
#             D[:, j] = (season_idx == j).astype(float)

#         # X = [1, t, D1..D_{S-1}]
#         X = np.column_stack([np.ones(t), tpast, D])

#         beta, _, _, _ = np.linalg.lstsq(X, y_past, rcond=None)

#         mu = beta[0]
#         b = beta[1]
#         gam = beta[2:]                           # length S-1

#         # Reconstruct full seasonal effects with sum-to-zero constraint
#         full_g = np.zeros(S, dtype=float)
#         full_g[:S - 1] = gam
#         full_g[S - 1] = -np.sum(gam)

#         a_hat[t] = mu
#         b_hat[t] = b
#         g_hat[t, :] = full_g

#         t_next = t + 1                           # next time index in 1-based scale
#         season_next = (t % S)                    # season for time t_next (0-based)
#         yhat[t] = mu + b * t_next + full_g[season_next]

#     return yhat, a_hat, b_hat, g_hat




def running_seasonal_regression_forecast(y, S):
    """
    Running Seasonal Regression using:
        δ̂_J = (Σ x x')^{-1} (Σ x y)

    x_t' = (1, t, dummies) where:
      - if season in 0..S-2: dummy one-hot
      - if season == S-1 (last): dummy all -1
    """
    y = np.asarray(y, dtype=float)
    T = len(y)

    k = S + 1  # parameters: mu, beta, gamma_1..gamma_{S-1}
    yhat = np.full(T, np.nan)
    a_hat = np.full(T, np.nan)
    b_hat = np.full(T, np.nan)
    g_hat = np.full((T, S), np.nan)

    def x_vec(t_index_0based):
        t1 = float(t_index_0based + 1)       # 1-based time
        season = t_index_0based % S          # 0..S-1
        d = np.zeros(S - 1, dtype=float)
        if season == S - 1:
            d[:] = -1.0                      # last season encoding 
        else:
            d[season] = 1.0                  # one-hot
        return np.concatenate(([1.0, t1], d))

    # Accumulators: A = Σ x x', b = Σ x y
    A = np.zeros((k, k), dtype=float)
    b = np.zeros(k, dtype=float)

    # We need at least k observations to invert A stably.
    # We'll start forecasts at t = k (0-based), i.e., using data up to t-1.
    # (This matches your "S+1" idea: k = S+1)
    for t in range(T):
        # Forecast y[t] using δ̂ built from past data (0..t-1)
        if t >= k and np.linalg.matrix_rank(A) == k:
            delta = np.linalg.solve(A, b)  # δ̂_{t-1}
            yhat[t] = x_vec(t) @ delta

            mu = delta[0]
            beta = delta[1]
            gam = delta[2:]  # length S-1
            full_g = np.zeros(S, dtype=float)
            full_g[:S-1] = gam
            full_g[S-1] = -np.sum(gam)
            a_hat[t] = mu
            b_hat[t] = beta
            g_hat[t, :] = full_g

        # Now incorporate current observation into A,b for future steps
        x = x_vec(t)
        A += np.outer(x, x)
        b += x * y[t]

    return yhat, a_hat, b_hat, g_hat


# ============================================================
# 4) Holt–Winters Seasonal (Additive / Multiplicative)
# ============================================================

def _hw_init(y, S, multiplicative):
    y = np.asarray(y, dtype=float)
    if len(y) < 2 * S:
        raise ValueError("Need at least 2 full seasons (2S observations) for initialization.")

    L_S = np.mean(y[:S])
    next_season_mean = np.mean(y[S:2 * S])
    G_S = (next_season_mean - L_S) / S

    if multiplicative:
        H = y[:S] / L_S
    else:
        H = y[:S] - L_S

    return L_S, G_S, H


def holt_winters_additive_forecast(y, S, alpha, beta, gamma):
    """
    Holt–Winters additive seasonality one-step-ahead forecast series.

    Returns fitted one-step-ahead forecasts yhat (NaN where undefined),
    plus level, growth, and seasonal arrays.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)

    L, G, H0 = _hw_init(y, S, multiplicative=False)

    L_t = np.full(T, np.nan)
    G_t = np.full(T, np.nan)
    H_t = np.full(T, np.nan)
    yhat = np.full(T, np.nan)

    # seed at index S-1 representing time S (1-based)
    L_t[S - 1] = L
    G_t[S - 1] = G
    H_t[:S] = H0

    for t in range(S, T):
        # forecast y[t] using states at t-1
        yhat[t] = (L_t[t - 1] + G_t[t - 1]) + H_t[t - S]

        # updates using y[t]
        L_new = alpha * (y[t] - H_t[t - S]) + (1 - alpha) * (L_t[t - 1] + G_t[t - 1])
        G_new = beta * (L_new - L_t[t - 1]) + (1 - beta) * G_t[t - 1]
        H_new = gamma * (y[t] - L_new) + (1 - gamma) * H_t[t - S]

        L_t[t] = L_new
        G_t[t] = G_new
        H_t[t] = H_new

    return yhat, L_t, G_t, H_t


def holt_winters_multiplicative_forecast(y, S, alpha, beta, gamma):
    """
    Holt–Winters multiplicative seasonality one-step-ahead forecast series.

    Returns fitted one-step-ahead forecasts yhat (NaN where undefined),
    plus level, growth, and seasonal arrays.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)

    L, G, H0 = _hw_init(y, S, multiplicative=True)

    L_t = np.full(T, np.nan)
    G_t = np.full(T, np.nan)
    H_t = np.full(T, np.nan)
    yhat = np.full(T, np.nan)

    L_t[S - 1] = L
    G_t[S - 1] = G
    H_t[:S] = H0

    for t in range(S, T):
        # forecast y[t] using states at t-1
        yhat[t] = (L_t[t - 1] + G_t[t - 1]) * H_t[t - S]

        # updates using y[t]
        L_new = alpha * (y[t] / H_t[t - S]) + (1 - alpha) * (L_t[t - 1] + G_t[t - 1])
        G_new = beta * (L_new - L_t[t - 1]) + (1 - beta) * G_t[t - 1]
        H_new = gamma * (y[t] / L_new) + (1 - gamma) * H_t[t - S]

        L_t[t] = L_new
        G_t[t] = G_new
        H_t[t] = H_new

    return yhat, L_t, G_t, H_t


def estimate_alpha_beta_gamma_seasonal_hw(y, S, multiplicative=False, criterion="SSE", grid_n=31, eps=1e-2):
    """
    Expanding-window Seasonal Holt–Winters with re-estimated (alpha,beta,gamma) at each step.

    Returns
    -------
    alpha_t, beta_t, gamma_t : np.ndarray
        Parameter chosen for each time t (NaN where undefined).
    yhat_t : np.ndarray
        One-step-ahead forecast series (NaN where undefined).
    u_t : np.ndarray
        One-step-ahead errors y - yhat (NaN where undefined).
    loss_t : np.ndarray
        Best loss at each time.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)

    grid = np.linspace(eps, 1.0 - eps, grid_n)

    alpha_t = np.full(T, np.nan)
    beta_t  = np.full(T, np.nan)
    gamma_t = np.full(T, np.nan)
    yhat_t  = np.full(T, np.nan)
    u_t     = np.full(T, np.nan)
    loss_t  = np.full(T, np.nan)

    # Need at least 2 full seasons for _hw_init in helper.py => len(y_sub) >= 2S
    # And need t >= 2S to forecast index t (since y_sub length = t)
    for t in range(2 * S, T):
        y_sub = y[:t]  # history up to t-1, used to forecast y[t]

        best_a = best_b = best_g = None
        best_loss = np.inf

        for a in grid:
            for b in grid:
                for g in grid:
                    if multiplicative:
                        F, _, _, _ = holt_winters_multiplicative_forecast(y_sub, S, a, b, g)
                    else:
                        F, _, _, _ = holt_winters_additive_forecast(y_sub, S, a, b, g)

                    # evaluate fitted one-step-ahead residuals inside y_sub
                    idx = np.arange(S, len(y_sub))
                    resid = y_sub[idx] - F[idx]

                    if criterion == "ME":
                        loss = float(np.mean(resid))
                    elif criterion == "MAE":
                        loss = float(np.mean(np.abs(resid)))
                    elif criterion == "MAPE":
                        denom = np.where(y_sub[idx] == 0, np.nan, np.abs(y_sub[idx]))
                        loss = float(np.nanmean(100.0 * np.abs(resid) / denom))
                    elif criterion == "MSE":
                        loss = float(np.mean(resid ** 2))
                    elif criterion == "SSE":
                        loss = float(np.sum(resid ** 2))
                    else:
                        raise ValueError("criterion must be one of: 'ME','MAE','MAPE','MSE','SSE'")

                    if loss < best_loss:
                        best_loss = loss
                        best_a, best_b, best_g = a, b, g

        # forecast y[t] using best params fitted on y[:t]
        if multiplicative:
            F_best, _, _, _ = holt_winters_multiplicative_forecast(y_sub, S, best_a, best_b, best_g)
        else:
            F_best, _, _, _ = holt_winters_additive_forecast(y_sub, S, best_a, best_b, best_g)

        # one-step-ahead forecast for index t is the last fitted forecast in y_sub
        yhat_t[t] = F_best[-1]
        u_t[t] = y[t] - yhat_t[t]

        alpha_t[t] = float(best_a)
        beta_t[t]  = float(best_b)
        gamma_t[t] = float(best_g)
        loss_t[t]  = float(best_loss)

    return alpha_t, beta_t, gamma_t, yhat_t, u_t, loss_t


# =============================
# Errors and evaluation metrics
# =============================

def forecast_errors(y, yhat):
    """Compute one-step-ahead forecast errors."""
    n = len(y)                            # Length of series
    u = [None] * n                        # Initialize error list

    for t in range(1, n):                 # Loop from second observation
        u[t] = float(y[t]) - float(yhat[t])  # Compute error

    return u                              # Return errors


def forecast_metrics(u, y, tau=1):
    """Compute ME, MAE, MAPE, and MSE from t=tau onward."""
    u = np.asarray(u, dtype=float)[tau-1:]  # Slice errors
    y = np.asarray(y, dtype=float)[tau-1:]  # Slice actuals

    me = u.mean()                         # Mean error
    mae = np.abs(u).mean()                # Mean absolute error
    mape = (100.0 * np.abs(u) / np.abs(y)).mean()  # MAPE
    mse = (u ** 2).mean()                 # Mean squared error

    return {"ME": me, "MAE": mae, "MAPE": mape, "MSE": mse}  # Return dict


def metrics(y_true, y_hat):
    """Return ME, MAE, MAPE, MSE as a tuple."""
    u = y_true - y_hat                    # Forecast errors
    me = float(np.mean(u))                # Mean error
    mae = float(np.mean(np.abs(u)))       # Mean absolute error
    mape = float(np.mean(100.0 * np.abs(u) / np.abs(y_true)))  # MAPE
    mse = float(np.mean(u**2))            # Mean squared error
    return me, mae, mape, mse              # Return tuple


def metrics_row(y, yhat, start_idx):
    """Return metrics dict for sliced series starting at start_idx."""
    me, mae, mape, mse = metrics(y[start_idx:], yhat[start_idx:])  # Compute metrics
    return {"ME": me, "MAE": mae, "MAPE": mape, "MSE": mse}         # Return dict



def metrics_last_k(y, u, k):
    """
    Compute forecast error metrics over the last k observations.
 
    """
    # Extract the last k residuals
    u_last = u[-k:]

    # Extract the last k true observations
    y_last = y[-k:]

    # Compute forecast metrics using a helper function with a horizon of 1
    metrics_dict = forecast_metrics(u_last, y_last, tau=1)

    # Return the metrics in a fixed order as a tuple
    return (
        metrics_dict["ME"],
        metrics_dict["MAE"],
        metrics_dict["MAPE"],
        metrics_dict["MSE"]
    )