"""
Script for performing the Volatility Targeting Backtest using the GARCH-Copula-XGBoost framework

"""

import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# Config

DATA_DIR = Path("Data/Verified")
RESULTS_DIR = Path("Results/VolTargeting")
PLOT_DIR = Path("Plots/VolTargeting") 
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTOS = ["BTC", "ETH", "BNB", "XRP"]
STABLES = ["USDT", "USDC", "DAI"]

FACTOR_CONFIG = {
    'Crypto_Up':   (CRYPTOS, 'Upside_Vol'),
    'Crypto_Down': (CRYPTOS, 'Downside_Vol'),
    'Stable_Vol':  (STABLES, 'LogVolChange'),
    'Stable_Up':   (STABLES, 'Upside_Vol'),
    'Stable_Down': (STABLES, 'Downside_Vol')
}

START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01'
FULL_END_DATE = '2025-01-01'

VOL_TARGETS = [0.20, 0.30, 0.40, 0.50] 

TRADING_DAYS = 366
RETRAIN_DAYS = 30           
BPS_COST = 0.0001           
WIN_LIMITS = (0.01, 0.01)   
MAX_LAGS = 1
LAGS = [1,7,30]
GARCH = 'EGARCH'
ERROR_DIST = 'skewt'
SEED = 123
Z_SCORE_WINDOW = 60

PARAM_GRID = {
    'n_estimators': [50, 100],                
    'learning_rate': [0.05, 0.1],             
    'max_depth': [3, 4, 5],                     
    'subsample': [0.7, 0.8],                   
    'colsample_bytree': [0.7, 0.8],               
    'reg_alpha': [0, 0.05],                   
    'reg_lambda': [0.5, 1.5]                  
}


def get_data_dict():

    data_dict = {}

    for file in DATA_DIR.glob("*.csv"):

        coin = file.stem.replace("Verif_", "") 

        df = pd.read_csv(file)

        if 'Date' in df.columns:

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            data_dict[coin] = df

    print(f"Loaded {len(data_dict)} assets: {list(data_dict.keys())}")

    return data_dict

def calculate_pca_for_window(coins, var, data_dict, current_end_date):

    current_date = pd.to_datetime(current_end_date)
    yesterday = current_date - pd.Timedelta(days=1)

    train_start = pd.to_datetime(START_DATE)
    
    df_list = [data_dict[c][var] for c in coins if c in data_dict and var in data_dict[c].columns]

    if not df_list: 
        return pd.Series(dtype=float), None, None
    
    raw_df = pd.concat(df_list, axis=1, keys=coins).dropna()
    
    train_data = raw_df.loc[train_start:yesterday]
    test_data = raw_df.loc[[current_date]] if current_date in raw_df.index else pd.DataFrame()
    
    if train_data.empty: 
        return pd.Series(dtype=float), None, None

    lower = train_data.quantile(WIN_LIMITS[0])
    upper = train_data.quantile(1 - WIN_LIMITS[1])
    train_data = train_data.clip(lower=lower, upper=upper, axis=1)
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    
    pca = PCA(n_components=1)
    train_factor = pca.fit_transform(train_scaled)
    
    flipped = False
    if 'BTC' in coins:

        if pca.components_[0][coins.index('BTC')] < 0: 
            flipped = True

    elif 'USDT' in coins:
        if pca.components_[0][coins.index('USDT')] < 0: 
            flipped = True

    elif np.sum(pca.components_[0]) < 0:
        flipped = True

    if flipped:
        pca.components_[0] *= -1
        train_factor *= -1

    loadings = pd.Series(pca.components_[0], index=coins)

    if not test_data.empty:
        test_data = test_data.clip(lower=lower, upper=upper, axis=1)
        test_scaled = scaler.transform(test_data)
        test_factor = np.dot(test_scaled, pca.components_.T) 

    else:
        test_factor = np.array([])

    full_vals = np.concatenate([train_factor.ravel(), test_factor.ravel() if not test_data.empty else []])
    full_idx = train_data.index.union(test_data.index)
    series = pd.Series(full_vals, index=full_idx)

    stats = {'mean': scaler.mean_.mean(), 'scale': scaler.scale_.mean()}
    
    return series, stats, loadings

def select_best_arma(series, max_order=MAX_LAGS):

    best_aic = np.inf
    best_order = (0, 0)

    for p in range(max_order + 1):
        for q in range(max_order + 1):
            if p == 0 and q == 0: continue
            try:
                model = ARIMA(series, order=(p, 0, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, q)
            except:
                continue
    return best_order

def fit_best_garch(series, p_arma, q_arma):
    try:
        am = arch_model(series, vol=GARCH, p=1, q=1, dist=ERROR_DIST, mean='AR', lags=p_arma)
        res = am.fit(disp='off', show_warning=False)
        return res
    except:
        return None

def apply_fixed_garch(series, params, p_arma, q_arma):
    try:
        am = arch_model(series, vol=GARCH, p=1, q=1, dist=ERROR_DIST, mean='AR', lags=p_arma)
        res = am.fix(params)
        return res
    except:
        return None

def get_volatility(garch_res):
    return garch_res.conditional_volatility

def transform_to_uniform(garch_res):

    std_resid = garch_res.std_resid

    dist = garch_res.model.distribution
    dist_params = [garch_res.params[p] for p in dist.parameter_names()]

    uniform_resid = dist.cdf(std_resid, dist_params)

    return pd.Series(uniform_resid, index=std_resid.index)

def get_forecast_mean(garch_res):
    return garch_res.forecast(horizon=1).mean.iloc[-1, 0]

def get_forecast_vol(garch_res):
    return np.sqrt(garch_res.forecast(horizon=1).variance.iloc[-1, 0])

def create_features(u_tgt, u_vol_tgt, src_dict):
    
    u_tgt = u_tgt.replace([np.inf, -np.inf], np.nan)
    u_vol_tgt = u_vol_tgt.replace([np.inf, -np.inf], np.nan)
    
    df = pd.DataFrame({
        'target_u': u_tgt,
        'target_vol': u_vol_tgt
    })
    
    ma_cols = [] 
    
    for s_name, data in src_dict.items():
    
        s_resid = data['resid'].replace([np.inf, -np.inf], np.nan)
        s_vol = data['vol'].replace([np.inf, -np.inf], np.nan)
        
        df[f'{s_name}_u'] = s_resid
        df[f'{s_name}_vol'] = s_vol
        
        df[f'{s_name}_u_ma_7'] = df[f'{s_name}_u'].rolling(window=7).mean().shift(1)
        df[f'{s_name}_vol_ma_7'] = df[f'{s_name}_vol'].rolling(window=7).mean().shift(1)
        
        df[f'{s_name}_u_ma_30'] = df[f'{s_name}_u'].rolling(window=30).mean().shift(1)
        df[f'{s_name}_vol_ma_30'] = df[f'{s_name}_vol'].rolling(window=30).mean().shift(1)
        
        ma_cols.extend([
            f'{s_name}_u_ma_7', f'{s_name}_vol_ma_7',
            f'{s_name}_u_ma_30', f'{s_name}_vol_ma_30'
        ])
    
    lags = LAGS
    cols_to_lag = [c for c in df.columns if '_ma_' not in c] 
    
    lag_cols = []
    for col in cols_to_lag:
        for lag in lags:
            col_name = f'{col}_lag{lag}'
            df[col_name] = df[col].shift(lag)
            lag_cols.append(col_name)
            
    df.dropna(inplace=True)
    
    target_col = 'target_u'
    y = df[target_col]
    
    valid_features = lag_cols + ['target_vol'] + ma_cols
    
    X = df[valid_features]
    
    bench_cols = [c for c in X.columns if 'target_' in c]
    X_bench = X[bench_cols]
    
    return X_bench, y, X, df.index


def save_feature_plots(models_cache, output_dir):
    
    print("\nGenerating Feature Importance and Gain Plots...")
    
    directions = ['Crypto_Up', 'Crypto_Down']
    
    # Setup Figures
    fig_imp, axes_imp = plt.subplots(1, 2, figsize=(18, 10))
    fig_gain, axes_gain = plt.subplots(1, 2, figsize=(12, 6))
    
    if isinstance(axes_imp, plt.Axes): axes_imp = [axes_imp]
    if isinstance(axes_gain, plt.Axes): axes_gain = [axes_gain]

    crypto_color = '#1f77b4' # Blue
    stable_color = '#ff7f0e' # Orange

    for idx, direction in enumerate(directions):
        if direction not in models_cache:
            continue
            
        model = models_cache[direction]['Chall']
        
        imp_dict = model.get_booster().get_score(importance_type='total_gain')
        
        if not imp_dict:
            print(f"No importance scores found for {direction}")
            continue

        sorted_feats = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
        top_n = min(20, len(sorted_feats))
        top_feats = sorted_feats[:top_n]
        
        feat_names = [x[0] for x in top_feats]
        feat_vals = [x[1] for x in top_feats]
        
        bar_colors = [stable_color if 'Stable' in n else crypto_color for n in feat_names]
        
        ax = axes_imp[idx]
        y_pos = np.arange(len(feat_names))
        ax.barh(y_pos, feat_vals, color=bar_colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_names)
        ax.invert_yaxis()
        ax.set_title(f'{direction} - Top {top_n} Features (Total Gain)')
        ax.set_xlabel('Total Gain')
        
        legend_elements = [mpatches.Patch(facecolor=crypto_color, label='Crypto Feature'),
                           mpatches.Patch(facecolor=stable_color, label='Stable Feature')]
        ax.legend(handles=legend_elements, loc='lower right')

        total_crypto_gain = 0.0
        total_stable_gain = 0.0
        
        for f_name, f_val in imp_dict.items():
            if 'Stable' in f_name:
                total_stable_gain += f_val
            else:
                total_crypto_gain += f_val
        
        grand_total = total_crypto_gain + total_stable_gain
        if grand_total > 0:
            pct_crypto = (total_crypto_gain / grand_total) * 100
            pct_stable = (total_stable_gain / grand_total) * 100
        else:
            pct_crypto = 0.0
            pct_stable = 0.0
        
        ax_g = axes_gain[idx]
        bars = ax_g.bar(['Crypto', 'Stable'], [total_crypto_gain, total_stable_gain], color=[crypto_color, stable_color])
        ax_g.set_title(f'{direction} - Aggregated Total Gain')
        ax_g.set_ylabel('Total Gain Sum')
        
        ax_g.text(0, total_crypto_gain, f'{total_crypto_gain:.1f}\n({pct_crypto:.1f}%)', 
                  ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax_g.text(1, total_stable_gain, f'{total_stable_gain:.1f}\n({pct_stable:.1f}%)', 
                  ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        y_max = max(total_crypto_gain, total_stable_gain)
        ax_g.set_ylim(0, y_max * 1.15)

    fig_imp.tight_layout()
    plot_path_imp = output_dir / "Challenger_Feature_Importance.png"
    fig_imp.savefig(plot_path_imp)
    
    fig_gain.tight_layout()
    plot_path_gain = output_dir / "Challenger_Total_Gains.png"
    fig_gain.savefig(plot_path_gain)
    
    print(f"Plots saved to: {output_dir}")

def run_strategy():
    print("1. Loading Data...")
    data_dict = get_data_dict()
    
    print("2. Pre-calculating factors & Selecting ARMA Orders...")
    factors_init = {}
    arma_orders = {}
    
    for name, (coins, col) in FACTOR_CONFIG.items():
        series, stats, loadings = calculate_pca_for_window(coins, col, data_dict, TRAIN_END_DATE)
        factors_init[name] = series
        
        p, q = select_best_arma(series.dropna(), max_order=MAX_LAGS)
        arma_orders[name] = (p, q)
        print(f"   {name}: Best ARMA Order = ({p},{q})")

    print("3. Initial Model Training...")
    
    best_params_cache = {
        'Crypto_Down': {'Bench': None, 'Chall': None},
        'Crypto_Up':   {'Bench': None, 'Chall': None}
    }

    def train_models(current_date, save_plots=False, use_cached_params=False, verbose=False):
        trained_models = {}
        source_names = [k for k in FACTOR_CONFIG.keys() if 'Stable' in k]
        tscv = TimeSeriesSplit(n_splits=3)
        
        def get_model(X, y, name_log, key_direction, key_model):
            if use_cached_params and best_params_cache[key_direction][key_model] is not None:
                params = best_params_cache[key_direction][key_model]
                model = xgb.XGBRegressor(random_state=SEED, n_jobs=-1, **params)
                model.fit(X, y)
                return model
            else:
                xgb_model = xgb.XGBRegressor(random_state=SEED, n_jobs=1) 
                gs = GridSearchCV(xgb_model, PARAM_GRID, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
                gs.fit(X, y)
                best_params_cache[key_direction][key_model] = gs.best_params_
                return gs.best_estimator_

        for direction in ['Crypto_Down', 'Crypto_Up']:
            train_end = current_date - pd.Timedelta(days=1)
            tgt_series, _, _ = calculate_pca_for_window(FACTOR_CONFIG[direction][0], FACTOR_CONFIG[direction][1], data_dict, train_end)
            p_tgt, q_tgt = arma_orders[direction]
            g_tgt_model = fit_best_garch(tgt_series, p_tgt, q_tgt)
            
            if g_tgt_model is None: continue

            u_tgt = transform_to_uniform(g_tgt_model)
            u_vol_tgt = get_volatility(g_tgt_model)
            
            src_data_dict = {}
            garch_objects = {} 
            for s_name in source_names:
                s_series, _, _ = calculate_pca_for_window(FACTOR_CONFIG[s_name][0], FACTOR_CONFIG[s_name][1], data_dict, train_end)
                p_src, q_src = arma_orders[s_name]
                g = fit_best_garch(s_series, p_src, q_src)
                if g: 
                    garch_objects[s_name] = g
                    src_data_dict[s_name] = {'resid': transform_to_uniform(g), 'vol': get_volatility(g)}
            
            X_b, y, X_c, _ = create_features(u_tgt, u_vol_tgt, src_data_dict)
            
            m_b = get_model(X_b, y, f"{direction} (Bench)", direction, 'Bench')
            m_c = get_model(X_c, y, f"{direction} (Chall)", direction, 'Chall')
            
            if verbose:
                print(f"\n[DEBUG] Feature Importances for {direction}:")
                if hasattr(m_c, 'feature_importances_'):
                    imps = pd.Series(m_c.feature_importances_, index=X_c.columns)
                    print(imps.sort_values(ascending=False).head(10))
                else:
                    print("Model does not support feature_importances_")
            
            trained_models[direction] = {
                'Bench': m_b, 'Chall': m_c, 
                'Garch_Params': g_tgt_model.params, 
                'Src_Params': {k: v.params for k, v in garch_objects.items()}, 
                'Garch_Model_Res': g_tgt_model,
                'Src_Model_Res': garch_objects 
            }
        return trained_models

    models_cache = train_models(pd.to_datetime(TRAIN_END_DATE), save_plots=False, use_cached_params=False, verbose=True)

    save_feature_plots(models_cache, PLOT_DIR)

    test_start_date = pd.to_datetime(TRAIN_END_DATE) + pd.Timedelta(days=1)
    full_dates = data_dict['BTC'].loc[test_start_date:FULL_END_DATE].index
    
    results = []
    
    prev_weights_b_map = {vt: {c: 0.0 for c in CRYPTOS} for vt in VOL_TARGETS}
    prev_weights_c_map = {vt: {c: 0.0 for c in CRYPTOS} for vt in VOL_TARGETS}
    prev_weights_n_map = {vt: {c: 0.0 for c in CRYPTOS} for vt in VOL_TARGETS}

    days_since_train = 0

    print(f"\n4. Running Walk-Forward Backtest ({len(full_dates)} days) across {len(VOL_TARGETS)} Vol Targets...")

    sig_history_b = []
    sig_history_c = []
    
    for t_date in tqdm(full_dates):
        
        days_since_train += 1
        if days_since_train >= RETRAIN_DAYS:
            models_cache = train_models(t_date, save_plots=False, use_cached_params=True, verbose=False)
            days_since_train = 0
            
        daily_factors = {}
        daily_stats = {}
        daily_loadings = {}
        
        for name, (coins, col) in FACTOR_CONFIG.items():
            series, stats, loadings = calculate_pca_for_window(coins, col, data_dict, t_date)
            daily_factors[name] = series
            daily_stats[name] = stats
            if loadings is not None: 
                daily_loadings[name] = loadings
        
        current_vol_levels = {'Up': [], 'Down': [], 'Total': []}
        prev_date = t_date - pd.Timedelta(days=1) 

        for c in CRYPTOS:
            if c in data_dict and prev_date in data_dict[c].index:
                row = data_dict[c].loc[prev_date] 
                
                if row['High'] > 0 and row['Low'] > 0 and row['Open'] > 0 and row['Close'] > 0:
                    log_h, log_l = np.log(row['High']), np.log(row['Low'])
                    log_o, log_c = np.log(row['Open']), np.log(row['Close'])
                    
                    rs_up = (log_h - log_c) * (log_h - log_o)
                    rs_down = (log_l - log_c) * (log_l - log_o)
                    
                    vol_up = np.sqrt(rs_up) if rs_up > 0 else 0
                    vol_down = np.sqrt(rs_down) if rs_down > 0 else 0
                    
                    current_vol_levels['Up'].append(vol_up)
                    current_vol_levels['Down'].append(vol_down)
                    current_vol_levels['Total'].append(vol_up + vol_down)
        
        avg_level_up = np.mean(current_vol_levels['Up']) if current_vol_levels['Up'] else 0.01
        avg_level_down = np.mean(current_vol_levels['Down']) if current_vol_levels['Down'] else 0.01
        
        naive_vol_daily_level = np.mean(current_vol_levels['Total']) if current_vol_levels['Total'] else 0.02
        naive_vol_ann = naive_vol_daily_level * np.sqrt(TRADING_DAYS)

        predictions_change = {} 
        targets = ['Crypto_Down', 'Crypto_Up']
        
        for direction in targets:
            if direction not in models_cache:
                predictions_change[direction] = {'Bench': 0.0, 'Chall': 0.0}
                continue
            
            model_data = models_cache[direction]
            
            series_tgt = daily_factors[direction]
            series_tgt_hist = series_tgt.loc[:t_date - pd.Timedelta(days=1)]
            p_tgt, q_tgt = arma_orders[direction]
            g_tgt_updated = apply_fixed_garch(series_tgt_hist, model_data['Garch_Model_Res'].params, p_tgt, q_tgt)
            if g_tgt_updated is None: continue

            mu_next = get_forecast_mean(g_tgt_updated)
            sigma_next = get_forecast_vol(g_tgt_updated)
            
            src_data_updated = {}
            for s_name, g_old in model_data['Src_Model_Res'].items():
                if g_old is None: continue
                s_series_hist = daily_factors[s_name].loc[:t_date - pd.Timedelta(days=1)]
                p_src, q_src = arma_orders[s_name]
                g_new = apply_fixed_garch(s_series_hist, g_old.params, p_src, q_src)
                if g_new:
                    src_data_updated[s_name] = {'resid': transform_to_uniform(g_new), 'vol': get_volatility(g_new)}

            dummy_idx = t_date
            u_tgt_new = pd.concat([transform_to_uniform(g_tgt_updated), pd.Series([0.5], index=[dummy_idx])])
            u_vol_new = pd.concat([get_volatility(g_tgt_updated), pd.Series([sigma_next], index=[dummy_idx])])
            src_new = {}
            for s_name, d in src_data_updated.items():
                src_new[s_name] = {
                    'resid': pd.concat([d['resid'], pd.Series([0.5], index=[dummy_idx])]),
                    'vol': pd.concat([d['vol'], pd.Series([0], index=[dummy_idx])])
                }

            X_b_next, _, X_c_next, _ = create_features(u_tgt_new, u_vol_new, src_new)
            
            model_b, model_c = model_data['Bench'], model_data['Chall']
            u_shock_b = model_b.predict(X_b_next.tail(1))[0]
            u_shock_c = model_c.predict(X_c_next.tail(1))[0]
            
            dist = g_tgt_updated.model.distribution
            params = [g_tgt_updated.params[p] for p in dist.parameter_names()]
            z_shock_b = dist.ppf(np.clip(u_shock_b, 0.001, 0.999), params)
            z_shock_c = dist.ppf(np.clip(u_shock_c, 0.001, 0.999), params)
            
            pred_raw_b = mu_next + (sigma_next * z_shock_b)
            pred_raw_c = mu_next + (sigma_next * z_shock_c)
            
            factor_hist_std = series_tgt_hist.std() if len(series_tgt_hist) > 0 else 1.0
            z_score_b = pred_raw_b / factor_hist_std
            z_score_c = pred_raw_c / factor_hist_std
            
            stats = daily_stats[direction]
            pred_change_b = (z_score_b * stats['scale']) + stats['mean']
            pred_change_c = (z_score_c * stats['scale']) + stats['mean']
            
            predictions_change[direction] = {'Bench': pred_change_b, 'Chall': pred_change_c}

        pred_change_down_b = predictions_change.get('Crypto_Down', {}).get('Bench', 0.0)
        pred_change_down_c = predictions_change.get('Crypto_Down', {}).get('Chall', 0.0)
        pred_change_up_b = predictions_change.get('Crypto_Up', {}).get('Bench', 0.0)
        pred_change_up_c = predictions_change.get('Crypto_Up', {}).get('Chall', 0.0)
        
        fcast_down_b = max(0.001, avg_level_down + pred_change_down_b)
        fcast_down_c = max(0.001, avg_level_down + pred_change_down_c)
        fcast_up_b = max(0.001, avg_level_up + pred_change_up_b)
        fcast_up_c = max(0.001, avg_level_up + pred_change_up_c)
        
        total_risk_daily_b = fcast_up_b + fcast_down_b
        total_risk_daily_c = fcast_up_c + fcast_down_c
        
        total_risk_ann_b = total_risk_daily_b * np.sqrt(TRADING_DAYS)
        total_risk_ann_c = total_risk_daily_c * np.sqrt(TRADING_DAYS)
        
        final_risk_ann_b = total_risk_ann_b
        final_risk_ann_c = total_risk_ann_c

        net_signal_b = (fcast_up_b - fcast_down_b) / (total_risk_daily_b)
        net_signal_c = (fcast_up_c - fcast_down_c) / (total_risk_daily_c)
        
        sig_history_b.append(net_signal_b)
        sig_history_c.append(net_signal_c)
        
        def get_z_multiplier(history, window=Z_SCORE_WINDOW):
            if len(history) < 2:
                return 1.0 
            data_slice = history[-window:]
            mu = np.mean(data_slice)
            sigma = np.std(data_slice)
            if sigma < 1e-6:
                return 1.0
            z_score = (history[-1] - mu) / sigma
            return 1 + np.tanh(z_score)

        skew_mult_b = get_z_multiplier(sig_history_b)
        skew_mult_c = get_z_multiplier(sig_history_c)
        
        loadings_up = daily_loadings.get('Crypto_Up')
        loadings_down = daily_loadings.get('Crypto_Down')
        
        if loadings_up is None or loadings_down is None:
            weights_ml = {c: 1/len(CRYPTOS) for c in CRYPTOS}
        else:
            L_up = loadings_up
            L_down = loadings_down
            ratio_scores = (L_up / L_down)
            if ratio_scores.sum() > 0:
                weights_ml = (ratio_scores / ratio_scores.sum()).to_dict()
            else:
                weights_ml = {c: 1/len(CRYPTOS) for c in CRYPTOS}

        weights_naive = {c: 1/len(CRYPTOS) for c in CRYPTOS}

        valid_sum_ml = 0.0
        valid_sum_naive = 0.0
        simple_ret_ml = 0.0
        simple_ret_naive = 0.0
        
        curr_vec_ml = {c: 0.0 for c in CRYPTOS}
        curr_vec_naive = {c: 0.0 for c in CRYPTOS}
        
        for coin in CRYPTOS:
            if coin in data_dict and t_date in data_dict[coin].index:
                r_log = data_dict[coin].loc[t_date]['Log Returns']
                r_simp = np.exp(r_log) - 1
                
                w_ml = weights_ml.get(coin, 0)
                simple_ret_ml += r_simp * w_ml
                valid_sum_ml += w_ml
                curr_vec_ml[coin] = w_ml

                w_n = weights_naive.get(coin, 0)
                simple_ret_naive += r_simp * w_n
                valid_sum_naive += w_n
                curr_vec_naive[coin] = w_n
        
        basket_ret_ml = np.log(1 + (simple_ret_ml / valid_sum_ml)) if valid_sum_ml > 0 else 0.0
        basket_ret_naive = np.log(1 + (simple_ret_naive / valid_sum_naive)) if valid_sum_naive > 0 else 0.0
        
        daily_result_row = {'Date': t_date, 'BuyHold': basket_ret_naive}
        
        for v_target in VOL_TARGETS:
            base_exp_b = min(1, v_target / final_risk_ann_b)
            base_exp_c = min(1, v_target / final_risk_ann_c)
            naive_exp = min(1, v_target / naive_vol_ann)
            
            total_exp_b = base_exp_b * skew_mult_b
            total_exp_c = base_exp_c * skew_mult_c
            
            cost_b, cost_c, cost_n = 0.0, 0.0, 0.0
            
            prev_w_b_vt = prev_weights_b_map[v_target]
            prev_w_c_vt = prev_weights_c_map[v_target]
            prev_w_n_vt = prev_weights_n_map[v_target]

            for c in CRYPTOS:
                r_c = data_dict[c].loc[t_date]['Log Returns'] if (c in data_dict and t_date in data_dict[c].index) else 0.0
                price_move = np.exp(r_c)
                
                val_b_pre = prev_w_b_vt.get(c, 0.0) * price_move
                tgt_b = total_exp_b * curr_vec_ml[c]
                cost_b += abs(tgt_b - val_b_pre) * BPS_COST
                prev_w_b_vt[c] = tgt_b
                
                val_c_pre = prev_w_c_vt.get(c, 0.0) * price_move
                tgt_c = total_exp_c * curr_vec_ml[c]
                cost_c += abs(tgt_c - val_c_pre) * BPS_COST
                prev_w_c_vt[c] = tgt_c
                
                val_n_pre = prev_w_n_vt.get(c, 0.0) * price_move
                tgt_n = naive_exp * curr_vec_naive[c]
                cost_n += abs(tgt_n - val_n_pre) * BPS_COST
                prev_w_n_vt[c] = tgt_n

            daily_result_row[f'Benchmark_{v_target}'] = (basket_ret_ml * total_exp_b) - cost_b
            daily_result_row[f'Challenger_{v_target}'] = (basket_ret_ml * total_exp_c) - cost_c
            daily_result_row[f'Naive_{v_target}'] = (basket_ret_naive * naive_exp) - cost_n

        results.append(daily_result_row)
        
    res_df = pd.DataFrame(results).set_index('Date')
    res_df.to_csv(RESULTS_DIR / "Backtest_Results.csv")
    
    print("\n=== Final Strategy Performance (Multi-Vol) ===")
    
    metrics_list = []
    strat_cols = [c for c in res_df.columns if c != 'BuyHold']
    
    r_bh = res_df['BuyHold']
    ann_ret_bh = r_bh.mean() * TRADING_DAYS
    ann_vol_bh = r_bh.std() * np.sqrt(TRADING_DAYS)
    down_dev_bh = r_bh[r_bh<0].std() * np.sqrt(TRADING_DAYS)
    sortino_bh = ann_ret_bh / down_dev_bh if down_dev_bh > 0 else 0
    mdd_bh = (1+r_bh).cumprod() / (1+r_bh).cumprod().cummax() - 1
    metrics_list.append({
        'Strategy': 'BuyHold', 'Vol_Target': 'N/A',
        'Ann_Return': ann_ret_bh, 'Ann_Vol': ann_vol_bh, 
        'Sortino': sortino_bh, 'MaxDD': mdd_bh.min()
    })

    for col in strat_cols:
        parts = col.split('_')
        strat_name = parts[0]
        vol_level = parts[1]
        
        r = res_df[col]
        ann_ret = r.mean() * TRADING_DAYS
        ann_vol = r.std() * np.sqrt(TRADING_DAYS)
        down_dev = r[r<0].std() * np.sqrt(TRADING_DAYS)
        sortino = ann_ret / down_dev if down_dev > 0 else 0
        mdd_series = (1+r).cumprod() / (1+r).cumprod().cummax() - 1
        max_dd = mdd_series.min()
        
        metrics_list.append({
            'Strategy': strat_name,
            'Vol_Target': vol_level,
            'Ann_Return': ann_ret,
            'Ann_Vol': ann_vol,
            'Sortino': sortino,
            'MaxDD': max_dd
        })

    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = RESULTS_DIR / "Performance_Metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    print(metrics_df.to_string())
    print(f"\nSaved Multi-Level Results to: {RESULTS_DIR}")

if __name__ == "__main__":
    run_strategy()