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

# FOR NN IT DOES NOT HAVE PATIENCE PARAMETER AND DOES NOT COMPARE TO VALIDATION ERROR TO AVOID OVERFITTING YET
def bridge_pytorch(model_type, params, X, y, **kwargs):
    """Bridge for Neural Networks."""
    # Convert y from -1/1 to 0/1 for CrossEntropy
    y_mapped = ((y + 1) / 2).astype(int)
    
    # Reshape X to 3D: (Samples, 1, Features) - Single-bar 'sequence' for simplicity
    X_3d = torch.FloatTensor(X.values.copy()).unsqueeze(1)
    y_tensor = torch.LongTensor(y_mapped.values.copy())
    
    model = TimeSeriesNet(X.shape[1], params['hidden_dim'], params['n_layers'], model_type)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10): # Example epoch count
        optimizer.zero_grad()
        output = model(X_3d)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    
    return model

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
        'bridge': lambda c, p, X, y, **k: bridge_pytorch('LSTM', p, X, y),
        'class': 'LSTM',
        'predictor': lambda m, X: (torch.argmax(m(torch.FloatTensor(X.values.copy()).unsqueeze(1)), dim=1).numpy() * 2 - 1),
        'prob_predictor': lambda m, X: torch.softmax(m(torch.FloatTensor(X.values.copy()).unsqueeze(1)), dim=1)[:, 1].detach().numpy(),
        'suggest': lambda t: {'hidden_dim': t.suggest_int('m1__hidden_dim', 16, 128), 'n_layers': t.suggest_int('m1__n_layers', 1, 3), 'lr': t.suggest_float('m1__lr', 1e-4, 1e-2)}
    },
    'GRU': {
        'bridge': lambda c, p, X, y, **k: bridge_pytorch('GRU', p, X, y),
        'class': 'GRU',
        'predictor': lambda m, X: (torch.argmax(m(torch.FloatTensor(X.values.copy()).unsqueeze(1)), dim=1).numpy() * 2 - 1),
        'prob_predictor': lambda m, X: torch.softmax(m(torch.FloatTensor(X.values.copy()).unsqueeze(1)), dim=1)[:, 1].detach().numpy(),
        'suggest': lambda t: {'hidden_dim': t.suggest_int('m1__hidden_dim', 16, 128), 'n_layers': t.suggest_int('m1__n_layers', 1, 3), 'lr': t.suggest_float('m1__lr', 1e-4, 1e-2)}
    }
}
