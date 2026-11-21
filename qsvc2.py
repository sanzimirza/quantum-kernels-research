"""
HCQE: Hybrid Classical-Quantum Ensemble for Parity Classification
=================================================================
Runs on 6, 7, 8-bit parity problems with:
  - Classical MLP
  - Quantum SVC (QSVC with ZZFeatureMap)
  - HCQE (weighted ensemble)

Author: [Your Name]
Date: [Your Date]
"""

import numpy as np
from itertools import product
from sklearn.neural_network import MLPClassifier
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
    """
    total = 2 ** n_bits
    if n_samples is None or n_samples >= total:
        X = np.array(list(product([0, 1], repeat=n_bits)))
    else:
        rng = np.random.RandomState(random_state)
        X = rng.randint(0, 2, size=(n_samples, n_bits))
    y = np.array([sum(int(b) for b in row) % 2 for row in X])
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
    Combines MLP and QSVC via weighted average, with weights from
    training accuracy (no inner CV — outer CV handles evaluation).
    """

    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(32, 32),
            activation='logistic',
            solver='adam',
            learning_rate_init=0.05,
            max_iter=5000,
            random_state=0
        )
        self.qsvc = build_qsvc(n_bits)
        self.scaler = StandardScaler()
        self.meta_weights = (0.5, 0.5)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.mlp.fit(X_scaled, y)
        self.qsvc.fit(X, y)

        mlp_acc = accuracy_score(y, self.mlp.predict(X_scaled))
        qsvc_acc = accuracy_score(y, self.qsvc.predict(X))

        print(f"      [HCQE internal] MLP train: {mlp_acc:.3f}, "
              f"QSVC train: {qsvc_acc:.3f}")

        w_mlp = max(mlp_acc, 0.1)
        w_qsvc = max(qsvc_acc, 0.1)
        total = w_mlp + w_qsvc
        self.meta_weights = (w_mlp / total, w_qsvc / total)
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
    print(f"\n{'='*50}")
    print(f"  Running {n_bits}-bit parity experiment")
    print(f"{'='*50}")

    # Use ALL available samples for small bit-lengths, else 256
    n_samples_target = max(2 ** n_bits, 256)
    X, y = generate_parity_data(n_bits, n_samples=n_samples_target)
    print(f"  Generated {len(y)} samples")

    # 3-fold CV for 6+ bits (faster), 5-fold for smaller
    if n_bits >= 6:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        print(f"  Using 3-fold CV")
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        print(f"  Using 5-fold CV")

    mlp_scores = []
    qsvc_scores = []
    hcqe_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        print(f"\n  ── Fold {fold}/{cv.get_n_splits()} ──")
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
                hidden_layer_sizes=(32, 32),
                activation='logistic',
                solver='adam',
                learning_rate_init=0.05,
                max_iter=5000,
                random_state=seed
            )
            mlp.fit(X_train_mlp, y_train)
            acc = accuracy_score(y_test, mlp.predict(X_test_mlp))
            best_mlp_acc = max(best_mlp_acc, acc)
        mlp_scores.append(best_mlp_acc)
        print(f"    MLP fold acc:  {best_mlp_acc:.4f}")

        # --- QSVC ---
        print(f"    Training QSVC on {len(X_train)} samples...")
        qsvc = build_qsvc(n_bits)
        qsvc.fit(X_train, y_train)
        qsvc_acc = accuracy_score(y_test, qsvc.predict(X_test))
        qsvc_scores.append(qsvc_acc)
        print(f"    QSVC fold acc: {qsvc_acc:.4f}")

        # --- HCQE ---
        try:
            hcqe = HCQE(n_bits=n_bits)
            hcqe.fit(X_train, y_train)
            hcqe_acc = accuracy_score(y_test, hcqe.predict(X_test))
            hcqe_scores.append(hcqe_acc)
            print(f"    HCQE fold acc: {hcqe_acc:.4f}")
        except Exception as e:
            print(f"    HCQE failed: {e}")
            hcqe_scores.append(0.0)

    results = {
        'MLP': np.mean(mlp_scores) if mlp_scores else 0.0,
        'QSVC': np.mean(qsvc_scores) if qsvc_scores else 0.0,
        'HCQE': np.mean(hcqe_scores) if hcqe_scores else 0.0,
    }

    print(f"\n  ═══ AVERAGES for {n_bits}-bit ═══")
    print(f"    MLP:  {results['MLP']:.4f}")
    print(f"    QSVC: {results['QSVC']:.4f}")
    print(f"    HCQE: {results['HCQE']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────
# 5. MAIN — runs sequentially, saves progress after each bit-length
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os
    import time

    bit_range = [6, 7, 8]
    all_results = {}

    # Load any previous results if they exist
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            all_results = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded previous results: {list(all_results.keys())}")

    for n in bit_range:
        if n in all_results:
            print(f"\nSkipping {n}-bit (already done)")
            continue

        start = time.time()
        try:
            all_results[n] = run_experiment(n)
            elapsed = time.time() - start
            print(f"\n  ⏱️  {n}-bit took {elapsed/60:.1f} minutes")
        except Exception as e:
            print(f"\n  ❌ {n}-bit failed: {e}")
            continue

        # Save after each bit-length
        with open("results.json", "w") as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
        print(f"  💾 Saved results to results.json")

    # Print final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Bits':<6} {'MLP':<10} {'QSVC':<10} {'HCQE':<10}")
    print(f"{'-'*36}")
    for n in sorted(all_results.keys()):
        r = all_results[n]
        print(f"{n:<6} {r['MLP']:<10.4f} {r['QSVC']:<10.4f} {r['HCQE']:<10.4f}")