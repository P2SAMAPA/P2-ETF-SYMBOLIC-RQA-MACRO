import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def compute_macro_weights(macro_df, target_returns):
    """
    Estimate weights for macro variables via ridge regression of next-day return on macro.
    Returns: weight vector (normalised) and scaler for macro variables.
    """
    if len(macro_df) != len(target_returns):
        min_len = min(len(macro_df), len(target_returns))
        macro_df = macro_df.iloc[:min_len]
        target_returns = target_returns[:min_len]
    if len(target_returns) < 5:
        return np.ones(macro_df.shape[1]) / macro_df.shape[1], StandardScaler()
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(macro_scaled, target_returns)
    coef = ridge.coef_
    # Normalise weights to positive and sum to 1
    pos_weights = np.maximum(coef, 0)
    if pos_weights.sum() == 0:
        pos_weights = np.ones_like(coef) / len(coef)
    else:
        pos_weights = pos_weights / pos_weights.sum()
    return pos_weights, scaler

def macro_factor(macro_row, weights, scaler):
    """Compute composite macro factor from a single row of macro variables."""
    # Standardise macro row using the same scaler (if fit)
    if scaler is not None:
        macro_scaled = scaler.transform(macro_row.reshape(1, -1)).flatten()
    else:
        macro_scaled = macro_row
    # Weighted sum (positive weights only)
    factor = np.dot(weights, macro_scaled)
    # Exponential ensures positivity
    return np.exp(factor)

def recurrence_rate(series, macro_factor, embedding_dim=3, tau=1, base_quantile=0.1, gamma=0.5):
    """
    Compute recurrence rate for the last point in the series using macro‑conditioned threshold.
    """
    n = len(series)
    if n < embedding_dim * tau + 2:
        return 0.0
    # State space reconstruction
    N_state = n - (embedding_dim - 1) * tau
    states = np.zeros((N_state, embedding_dim))
    for i in range(N_state):
        for j in range(embedding_dim):
            states[i, j] = series[i + j * tau]
    # Compute pairwise distances (only for the last point? We need threshold for the whole recurrence matrix)
    # We'll use the global distribution of distances to set base threshold
    all_distances = []
    for i in range(N_state):
        for j in range(N_state):
            if i != j:
                dist = np.linalg.norm(states[i] - states[j])
                all_distances.append(dist)
    if len(all_distances) == 0:
        return 0.0
    base_threshold = np.percentile(all_distances, base_quantile * 100)
    # Adjust threshold by macro factor
    threshold = base_threshold * (1 + gamma * macro_factor)
    # For the last state (index N_state-1), count recurrences
    last_state = states[-1]
    recurrences = 0
    for i in range(N_state - 1):  # exclude self
        dist = np.linalg.norm(last_state - states[i])
        if dist < threshold:
            recurrences += 1
    # Recurrence rate = recurrences / (N_state - 1)
    rr = recurrences / (N_state - 1) if N_state > 1 else 0.0
    return rr

def rqa_score(returns, macro_df, embedding_dim=3, tau=1, base_quantile=0.1, gamma=0.5):
    """
    Compute RQA recurrence rate for a single ETF using macro‑conditioned threshold.
    Weights are estimated from the relationship between macro and next‑day returns (in‑sample within the window).
    """
    if len(returns) < embedding_dim * tau + 2 or macro_df is None or macro_df.empty:
        return 0.0
    # Align returns and macro
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Compute macro weights using ridge regression of next‑day return on macro (shifted by 1)
    # We need target_returns = returns[1:] and macro aligned
    if len(returns) < 2:
        return 0.0
    target = returns[1:]
    macro_aligned = macro_df.iloc[:-1] if len(macro_df) == len(returns) else macro_df[:len(target)]
    if len(target) != len(macro_aligned):
        min_len2 = min(len(target), len(macro_aligned))
        target = target[:min_len2]
        macro_aligned = macro_aligned.iloc[:min_len2]
    if len(target) < 5:
        return 0.0
    weights, scaler = compute_macro_weights(macro_aligned, target)
    # Current macro (last row) to compute factor for threshold
    current_macro = macro_df.iloc[-1].values
    factor = macro_factor(current_macro, weights, scaler)
    # Compute recurrence rate using macro‑adjusted threshold
    rr = recurrence_rate(returns, factor, embedding_dim, tau, base_quantile, gamma)
    return rr
