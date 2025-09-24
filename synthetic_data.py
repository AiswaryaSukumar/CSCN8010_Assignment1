# synthetic_data.py
import numpy as np
import pandas as pd

def make_synthetic_test(df_train, models, axes,
                        n_rows=None, sample_interval_sec=None,
                        anomaly_blocks=15, block_min_sec=10, block_max_sec=25,
                        drift_per_sec=0.0, seed=42,
                        residuals_dict=None,
                        force_above=True, MinC=None, MaxC=None):
    rng = np.random.default_rng(seed)
    
    # Estimate sampling interval if not given
    if sample_interval_sec is None or not np.isfinite(sample_interval_sec):
        diffs = df_train['time'].diff().dt.total_seconds()
        sample_interval_sec = float(np.nanmedian(diffs))
        if not np.isfinite(sample_interval_sec) or sample_interval_sec <= 0:
            sample_interval_sec = 1.0
    
    if n_rows is None:
        n_rows = len(df_train)
    
    # Create new time series after training period
    start_time = df_train['time'].max() + pd.to_timedelta(sample_interval_sec, unit="s")
    times = [start_time + pd.to_timedelta(i*sample_interval_sec, unit="s") for i in range(n_rows)]
    
    # time_numeric relative to new start
    t0 = times[0]
    time_numeric = np.array([(t - t0).total_seconds() for t in times], dtype=float)
    X_test = time_numeric.reshape(-1, 1)
    
    # Residual std per axis from training
    res_std = {a: float(np.std(residuals_dict[a])) for a in axes} if residuals_dict else {a: 1.0 for a in axes}
    
    # Baseline = model prediction + Gaussian noise (+ optional drift)
    data = {"time": times, "time_numeric": time_numeric}
    for axis in axes:
        yhat = models[axis].predict(X_test)
        noise = rng.normal(0.0, res_std[axis], size=n_rows)
        drift = drift_per_sec * time_numeric
        data[axis] = yhat + noise + drift
    
    df_test = pd.DataFrame(data)
    
    # Inject anomaly blocks
    # --- Inside synthetic_data.py, anomaly injection loop ---

    for _ in range(anomaly_blocks):
        block_len_sec = rng.uniform(block_min_sec, block_max_sec)
        block_len_rows = max(1, int(round(block_len_sec / sample_interval_sec)))
        start_idx = rng.integers(0, max(1, n_rows - block_len_rows))
        end_idx = min(n_rows, start_idx + block_len_rows)
        axis = rng.choice(axes)

    if force_above and MaxC is not None and MinC is not None:
        if rng.random() < 0.5:
            # Inject ALERT-level anomaly (between MinC and MaxC)
            lift = MinC[axis] * rng.uniform(1.1, 1.3)
        else:
            # Inject ERROR-level anomaly (above MaxC)
            lift = MaxC[axis] * rng.uniform(1.2, 1.5)
    else:
        lift = abs(res_std[axis]) * rng.uniform(2.5, 4.0)

    df_test.loc[start_idx:end_idx-1, axis] += lift

    return df_test, sample_interval_sec