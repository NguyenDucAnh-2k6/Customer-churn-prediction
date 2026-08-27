"""
PyTorch LSTM (Long Short-Term Memory) sequence classification model for Churn Prediction.
Implements Scikit-learn compatible estimator and Optuna model wrapper.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data.base import SplitResult
from src.models.base import BaseModelWrapper


class AttentionPooling(nn.Module):
    """Temporal attention pooling layer to dynamically weight timesteps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, hidden_dim)
        scores = self.attention(x)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(x * weights, dim=1)  # (batch_size, hidden_dim)
        return context


class LSTMChurnNet(nn.Module):
    """Deep PyTorch LSTM network with temporal attention and classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Normalization layer for features
        self.norm = nn.BatchNorm1d(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_mult = 2 if bidirectional else 1
        effective_hidden = hidden_dim * direction_mult

        if use_attention:
            self.pool = AttentionPooling(effective_hidden)
        else:
            self.pool = None

        self.classifier = nn.Sequential(
            nn.Linear(effective_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is 2D (batch_size, input_dim), reshape to (batch_size, 1, input_dim)
        if x.dim() == 2:
            x_norm = self.norm(x)
            x_seq = x_norm.unsqueeze(1)
        else:
            # If 3D (batch_size, seq_len, input_dim)
            b, s, d = x.shape
            x_flat = x.view(b * s, d)
            x_norm = self.norm(x_flat).view(b, s, d)
            x_seq = x_norm

        lstm_out, (hn, _) = self.lstm(x_seq)

        if self.use_attention and self.pool is not None:
            context = self.pool(lstm_out)
        else:
            if self.bidirectional:
                context = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
            else:
                context = hn[-1, :, :]

        logits = self.classifier(context)
        return logits.squeeze(-1)


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible PyTorch LSTM Classifier with sample weight and pos_weight support."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 25,
        scale_pos_weight: float = 1.0,
        patience: int = 5,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.scale_pos_weight = scale_pos_weight
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

        self.model_: Optional[LSTMChurnNet] = None
        self.scaler_ = StandardScaler()
        self.feature_names_: List[str] = []

    def _set_seed(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[Any, Any]]] = None,
    ) -> "LSTMClassifier":
        self._set_seed()

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            self.feature_names_ = [f"feat_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=np.float32)

        # Handle missing values
        X_arr = np.nan_to_num(X_arr, nan=0.0)

        # Fit Scaler
        X_scaled = self.scaler_.fit_transform(X_arr)

        input_dim = X_scaled.shape[1]
        self.model_ = LSTMChurnNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            use_attention=self.use_attention,
        ).to(self.device)

        # Prepare PyTorch Tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32)
        if sample_weight is not None:
            w_tensor = torch.tensor(sample_weight, dtype=torch.float32)
        else:
            w_tensor = torch.ones_like(y_tensor)

        train_dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Validation dataset if eval_set is provided
        val_loader = None
        if eval_set and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            if isinstance(X_val, pd.DataFrame):
                X_val_arr = X_val.values.astype(np.float32)
            else:
                X_val_arr = np.asarray(X_val, dtype=np.float32)
            X_val_arr = np.nan_to_num(X_val_arr, nan=0.0)
            X_val_scaled = self.scaler_.transform(X_val_arr)
            y_val_arr = np.asarray(y_val, dtype=np.float32)

            val_dataset = TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val_arr, dtype=torch.float32),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)

        # Optimizer, Loss, Scheduler
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        pos_weight_tensor = torch.tensor([self.scale_pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            total_loss = 0.0

            for batch_x, batch_y, batch_w in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_w = batch_w.to(self.device)

                optimizer.zero_grad()
                logits = self.model_(batch_x)
                losses = criterion(logits, batch_y)
                weighted_loss = torch.mean(losses * batch_w)
                weighted_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += weighted_loss.item() * len(batch_y)

            scheduler.step()

            # Validation step
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        v_logits = self.model_(vx)
                        v_losses = criterion(v_logits, vy)
                        val_loss += torch.mean(v_losses).item() * len(vy)
                val_loss /= len(val_dataset)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        self.model_.eval()
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        X_arr = np.nan_to_num(X_arr, nan=0.0)
        X_scaled = self.scaler_.transform(X_arr)

        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model_(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(probs)

        pos_probs = np.concatenate(probs_list, axis=0)
        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class LSTMModelWrapper(BaseModelWrapper):
    """Optuna Model Wrapper for PyTorch LSTM sequence neural network."""

    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        split_result: SplitResult,
        seed: int = 42,
    ) -> Dict[str, Any]:
        max_scale = max(3.0, min(15.0, split_result.scale_pos_weight_estimate * 1.1))

        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, max_scale),
            "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
            "use_attention": True,
            "epochs": 20,
            "patience": 5,
            "random_state": seed,
        }

    def build_model(
        self,
        params: Dict[str, Any],
        seed: int = 42,
    ) -> LSTMClassifier:
        p = params.copy()
        p.setdefault("random_state", seed)
        return LSTMClassifier(**p)

    def extract_feature_importances(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Estimate feature importance via input batch normalization weights."""
        if hasattr(model, "model_") and model.model_ is not None and hasattr(model.model_, "norm"):
            norm_weights = np.abs(model.model_.norm.weight.detach().cpu().numpy())
            df_imp = pd.DataFrame({"feature": feature_names, "importance": norm_weights})
            df_imp = df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
            return df_imp
        return pd.DataFrame({"feature": feature_names, "importance": np.ones(len(feature_names))})

    def save_model(self, model: Any, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(model, filepath)
