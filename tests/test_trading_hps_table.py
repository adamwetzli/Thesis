import random
from utils import generate_trading_hps_table

def get_synthetic_hps_data(folds=2):
    currencies = [
        'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 
        'USDCNH', 'USDJPY', 'USDMXN', 'USDTRY', 'USDZAR'
    ]
    params = [
        'sl_mult', 'tp_mult', 'risk_pct', 'kelly_fraction', 
        'mh', 'max_notional_exposure_pct', 'meta_significance'
    ]
    
    t_hps_list = []
    
    for f in range(folds):
        fold_data = {}
        for cur in currencies:
            for p in params:
                # Generate synthetic value
                if p == 'risk_pct':
                    val = round(random.uniform(0.005, 0.03), 4)
                elif p in ['sl_mult', 'tp_mult']:
                    val = round(random.uniform(1.0, 5.0), 2)
                elif p == 'mh':
                    val = float(random.randint(5, 50))
                else:
                    val = round(random.uniform(0.1, 1.0), 2)
                
                fold_data[f"{cur}_{p}"] = val
        t_hps_list.append(fold_data)
        
    return t_hps_list

# --- Execution ---
# Generate data for 2 folds
synthetic_data = get_synthetic_hps_data(folds=2)

# Call your function
generate_trading_hps_table(synthetic_data, "RandomForest", "test")