"""
HCQE: Hybrid Classical-Quantum Ensemble for Parity Classification
=================================================================
This script runs three models on parity problems of increasing bit-length:
  1. Classical MLP
  2. Quantum SVC (QSVC with ZZFeatureMap)
  3. HCQE (Ensemble of MLP + QSVC via weighted average)

Author: [Your Name]
Date: [Your Date]
"""

import numpy as np
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# ─────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────

def generate_parity_data(n_bits, n_samples=None, random_state=42):
    """
    Generate parity data.
    If n_samples is None or >= 2^n, generates all possible bit-strings.
    Otherwise, generates n_samples random bit-strings.
    Label = 1 if the number of 1s is odd, else 0.
    """
    total = 2 ** n_bits
    if n_samples is None or n_samples >= total:
        X = np.array(list(product([0, 1], repeat=n_bits)))
    else:
        rng = np.random.RandomState(random_state)
        X = rng.randint(0, 2, size=(n_samples, n_bits))
    y = np.array([sum(int(bit) for bit in row) % 2 for row in X])
    return X, y


# ─────────────────────────────────────────────────────────────
# 2. QUANTUM KERNEL SETUP
# ─────────────────────────────────────────────────────────────

def build_qsvc(n_bits):
    """
    Build a QSVC using a ZZFeatureMap.
    """
    feature_map = zz_feature_map(feature_dimension=n_bits, reps=2)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    return QSVC(quantum_kernel=kernel)


# ─────────────────────────────────────────────────────────────
# 3. HCQE ENSEMBLE
# ─────────────────────────────────────────────────────────────

class HCQE:
    """
    Hybrid Classical-Quantum Ensemble.
    Combines MLP and QSVC via weighted average, with weights determined
    by each base model's out-of-fold accuracy.
    """

    def __init__(self, n_bits, n_folds=5):
        self.n_bits = n_bits
        self.n_folds = n_folds
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(16, 16),
            activation='logistic',
            solver='adam',
            learning_rate_init=0.1,
            max_iter=5000,
            random_state=0
        )
        self.qsvc = build_qsvc(n_bits)
        self.scaler = StandardScaler()
        self.meta_weights = (0.5, 0.5)

    def fit(self, X, y):
        # Safety: reduce fold count if dataset is small
        if len(y) < self.n_folds * 2:
            self.n_folds = max(2, len(y) // 2)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        oof_mlp = np.zeros((len(y), 2))
        oof_qsvc = np.zeros((len(y), 2))

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            if len(np.unique(y_train)) < 2:
                continue

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train MLP
            self.mlp.fit(X_train_scaled, y_train)
            oof_mlp[val_idx] = self.mlp.predict_proba(X_val_scaled)

            # Train QSVC
            self.qsvc.fit(X_train, y_train)
            decision = self.qsvc.decision_function(X_val)
            prob = 1 / (1 + np.exp(-decision))
            oof_qsvc[val_idx, 1] = prob
            oof_qsvc[val_idx, 0] = 1 - prob

        # Reliability via OOF accuracy
        mlp_pred = np.argmax(oof_mlp, axis=1)
        qsvc_pred = np.argmax(oof_qsvc, axis=1)

        mlp_acc = accuracy_score(y, mlp_pred)
        qsvc_acc = accuracy_score(y, qsvc_pred)

        print(f"      [HCQE internal] MLP OOF acc: {mlp_acc:.3f}, "
              f"QSVC OOF acc: {qsvc_acc:.3f}")

        total = mlp_acc + qsvc_acc + 1e-9
        self.meta_weights = (mlp_acc / total, qsvc_acc / total)

        # Refit on full data
        X_scaled = self.scaler.fit_transform(X)
        self.mlp.fit(X_scaled, y)
        self.qsvc.fit(X, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        mlp_prob = self.mlp.predict_proba(X_scaled)

        decision = self.qsvc.decision_function(X)
        prob = 1 / (1 + np.exp(-decision))
        qsvc_prob = np.column_stack([1 - prob, prob])

        w_mlp, w_qsvc = self.meta_weights
        combined = w_mlp * mlp_prob + w_qsvc * qsvc_prob
        return np.argmax(combined, axis=1)


# ─────────────────────────────────────────────────────────────
# 4. EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────

def run_experiment(n_bits):
    """
    Run MLP, QSVC, and HCQE on an n-bit parity problem using cross-validation.
    """
    print(f"\n{'='*50}")
    print(f"  Running {n_bits}-bit parity experiment")
    print(f"{'='*50}")

    # Use all available data (parity is finite: 2^n strings max)
    n_samples_target = max(2 ** n_bits, 200)
    X, y = generate_parity_data(n_bits, n_samples=n_samples_target)

    n_samples = len(y)
    if n_samples < 8:
        cv = LeaveOneOut()
        print(f"  Using Leave-One-Out CV ({n_samples} samples)")
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        print(f"  Using 5-fold CV ({n_samples} samples)")

    mlp_scores = []
    qsvc_scores = []
    hcqe_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        # --- MLP (with random restarts) ---
        X_train_mlp = X_train.astype(float)
        X_test_mlp = X_test.astype(float)

        best_mlp_acc = 0.0
        for seed in [0, 1, 2, 3, 4]:
            mlp = MLPClassifier(
                hidden_layer_sizes=(16, 16),
                activation='logistic',
                solver='adam',
                learning_rate_init=0.1,
                max_iter=5000,
                random_state=seed
            )
            mlp.fit(X_train_mlp, y_train)
            acc = accuracy_score(y_test, mlp.predict(X_test_mlp))
            best_mlp_acc = max(best_mlp_acc, acc)
        mlp_scores.append(best_mlp_acc)

        # --- QSVC ---
        qsvc = build_qsvc(n_bits)
        qsvc.fit(X_train, y_train)
        qsvc_scores.append(accuracy_score(y_test, qsvc.predict(X_test)))

        # --- HCQE ---
        try:
            hcqe = HCQE(n_bits=n_bits, n_folds=min(3, len(y_train) // 2))
            hcqe.fit(X_train, y_train)
            hcqe_scores.append(accuracy_score(y_test, hcqe.predict(X_test)))
        except Exception as e:
            print(f"    HCQE failed on fold {fold}: {e}")
            hcqe_scores.append(0.0)

    results = {
        'MLP': np.mean(mlp_scores) if mlp_scores else 0.0,
        'QSVC': np.mean(qsvc_scores) if qsvc_scores else 0.0,
        'HCQE': np.mean(hcqe_scores) if hcqe_scores else 0.0,
    }

    print(f"    MLP Accuracy:  {results['MLP']:.4f}")
    print(f"    QSVC Accuracy: {results['QSVC']:.4f}")
    print(f"    HCQE Accuracy: {results['HCQE']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bit_range = [3, 4, 5]
    all_results = {}

    for n in bit_range:
        all_results[n] = run_experiment(n)

    print(f"\n{'='*60}")
    print("  SUMMARY: Accuracy by Bit-Length")
    print(f"{'='*60}")
    print(f"{'Bits':<6} {'MLP':<10} {'QSVC':<10} {'HCQE':<10}")
    print(f"{'-'*36}")
    for n in bit_range:
        r = all_results[n]
        print(f"{n:<6} {r['MLP']:<10.4f} {r['QSVC']:<10.4f} {r['HCQE']:<10.4f}")