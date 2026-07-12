import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ==============================================================================
# 1. PYTORCH NEURAL NETWORKS (LSTM, GRU, Transformer)
# ==============================================================================

class TimeSeriesNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, model_type='LSTM'):
        super().__init__()
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2) # Binary: Up or Down

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

def bridge_pytorch(model_type, params, X, y, max_epochs=100, patience=10, val_split=0.2,
                    seed=42, n_purged=0, n_embargo=0, **kwargs):
    """
    Bridge for Neural Networks.

    Trains with early stopping on a held-out validation split to avoid overfitting.
    The split is CHRONOLOGICAL (last `val_split` fraction of rows), never shuffled,
    since X/y are time-ordered bars and a random split would let the model "see"
    validation-adjacent information out of order and give an optimistic stopping signal.

    IMPORTANT: this carves a *new* train/val boundary inside whatever block of data
    it's given (e.g. inside a single inner-fold's already-purged training block).
    Labels are built from forward-looking windows (horizon/t_final/atr_lookback), so
    without a gap at this new boundary too, the tail of "train" could have labels
    informed by bars now sitting in "val" -- the same leakage n_purged/n_embargo
    prevent at every other fold boundary in this pipeline. `n_purged` rows are
    dropped off the end of the train slice and `n_embargo` rows off the start of
    the val slice to close that gap.

    Args:
        max_epochs (int): Hard ceiling on training epochs (early stopping usually
            triggers well before this).
        patience (int): Number of consecutive epochs without validation-loss
            improvement to tolerate before stopping.
        val_split (float): Fraction of the (already time-ordered) input reserved
            for validation / early-stopping monitoring, before the purge/embargo
            gap is removed.
        seed (int): Fixed torch seed so that repeated calls with identical
            hyperparameters produce identical weight initialization. Without this,
            Optuna's objective is noisy purely from random init, independent of the
            hyperparameters being tuned, which makes the HPO search much less efficient.
        n_purged (int): Rows dropped from the end of the internal train slice.
            Pass the same value used elsewhere in the pipeline (cockpit `n_purged`).
        n_embargo (int): Rows dropped from the start of the internal val slice.
            Pass the same value used elsewhere in the pipeline (cockpit `n_embargo`).
    """
    torch.manual_seed(seed)

    # Convert y from -1/1 to 0/1 for CrossEntropy
    y_mapped = ((y + 1) / 2).astype(int)

    # Chronological train/val split (no shuffling - this is time series data),
    # with a purge/embargo gap at the new internal boundary (see docstring above).
    n = len(X)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    train_end = n_train - n_purged
    val_start = n_train + n_embargo

    if train_end < 1 or val_start >= n:
        raise ValueError(
            f"Not enough rows ({n}) to carve out a val_split of {val_split} "
            f"with n_purged={n_purged} and n_embargo={n_embargo}."
        )

    X_train_raw, X_val_raw = X.iloc[:train_end], X.iloc[val_start:]
    y_train_raw, y_val_raw = y_mapped.iloc[:train_end], y_mapped.iloc[val_start:]

    # Reshape to 3D: (Samples, 1, Features) - Single-bar 'sequence' for simplicity
    X_train_t = torch.FloatTensor(X_train_raw.values.copy()).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train_raw.values.copy())
    X_val_t = torch.FloatTensor(X_val_raw.values.copy()).unsqueeze(1)
    y_val_t = torch.LongTensor(y_val_raw.values.copy())

    model = TimeSeriesNet(X.shape[1], params['hidden_dim'], params['n_layers'], model_type)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

        # Validation-loss check for early stopping
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = criterion(val_output, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Restore the weights from the epoch with the best validation loss,
    # not whatever the last (possibly overfit) epoch happened to produce.
    model.load_state_dict(best_state)
    model.eval()
    return model


def _nn_forward(model, X):
    """Shared inference path: eval mode + no_grad, so predictor/prob_predictor
    don't track gradients or leave the model in train mode after being called."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X.values.copy()).unsqueeze(1))


def nn_predictor(model, X):
    out = _nn_forward(model, X)
    return torch.argmax(out, dim=1).numpy() * 2 - 1


def nn_prob_predictor(model, X):
    out = _nn_forward(model, X)
    return torch.softmax(out, dim=1)[:, 1].numpy()

# ==============================================================================
# 2. THE EXPANDED MODEL REGISTRY
# ==============================================================================

MODELS = {
    # --- SKLEARN / GRADIENT BOOSTING ---
    'RandomForest': {
        'bridge': lambda c, p, X, y, **k: c(**p).fit(X, y),
        'class': RandomForestClassifier,
        'predictor': lambda m, X: m.predict(X),
        'prob_predictor': lambda m, X: m.predict_proba(X)[:, 1],
        'suggest': lambda t: {'n_estimators': t.suggest_int('m1__n_estimators', 50, 500), 'max_depth': t.suggest_int('m1__max_depth', 3, 20)}
    },
    'ExtraTrees': {
        'bridge': lambda c, p, X, y, **k: c(**p).fit(X, y),
        'class': ExtraTreesClassifier,
        'predictor': lambda m, X: m.predict(X),
        'prob_predictor': lambda m, X: m.predict_proba(X)[:, 1],
        'suggest': lambda t: {'n_estimators': t.suggest_int('m1__n_estimators', 50, 500), 'max_depth': t.suggest_int('m1__max_depth', 3, 20)}
    },
    'XGBoost': {
        'bridge': lambda c, p, X, y, **k: c(**p).fit(X, (y + 1) // 2), # Convert -1,1 → 0,1 for internal XGBoost representation
        'class': XGBClassifier,
        'predictor': lambda m, X: m.predict(X) * 2 - 1, # Convert 0,1 -> -1,1 for external use in the script
        'prob_predictor': lambda m, X: m.predict_proba(X)[:, 1],
        'suggest': lambda t: {'n_estimators': t.suggest_int('m1__n_estimators', 50, 300), 'learning_rate': t.suggest_float('m1__learning_rate', 0.01, 0.3)}
    },
    'LightGBM': {
        'bridge': lambda c, p, X, y, **k: c(**p).fit(X, y),
        'class': LGBMClassifier,
        'predictor': lambda m, X: m.predict(X),
        'prob_predictor': lambda m, X: m.predict_proba(X)[:, 1],
        'suggest': lambda t: {'n_estimators': t.suggest_int('m1__n_estimators', 50, 300), 'num_leaves': t.suggest_int('m1__num_leaves', 20, 100)}
    },

    # --- NEURAL NETWORKS ---
    'LSTM': {
        'bridge': lambda c, p, X, y, **k: bridge_pytorch('LSTM', p, X, y, n_purged=k.get('n_purged', 0), n_embargo=k.get('n_embargo', 0)),
        'class': 'LSTM',
        'predictor': nn_predictor,
        'prob_predictor': nn_prob_predictor,
        'suggest': lambda t: {'hidden_dim': t.suggest_int('m1__hidden_dim', 16, 128), 'n_layers': t.suggest_int('m1__n_layers', 1, 3), 'lr': t.suggest_float('m1__lr', 1e-4, 1e-2)}
    },
    'GRU': {
        'bridge': lambda c, p, X, y, **k: bridge_pytorch('GRU', p, X, y, n_purged=k.get('n_purged', 0), n_embargo=k.get('n_embargo', 0)),
        'class': 'GRU',
        'predictor': nn_predictor,
        'prob_predictor': nn_prob_predictor,
        'suggest': lambda t: {'hidden_dim': t.suggest_int('m1__hidden_dim', 16, 128), 'n_layers': t.suggest_int('m1__n_layers', 1, 3), 'lr': t.suggest_float('m1__lr', 1e-4, 1e-2)}
    }
}
