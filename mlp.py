"""
Minimal MLP parity test - isolates the MLP behavior.
"""
import numpy as np
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def generate_parity_data(n_bits):
    X = np.array(list(product([0, 1], repeat=n_bits)))
    y = np.array([sum(row) % 2 for row in X])
    return X, y

for n_bits in [3, 4, 5]:
    print(f"\n{'='*50}")
    print(f"  {n_bits}-bit parity")
    print(f"{'='*50}")

    X, y = generate_parity_data(n_bits)

    # Test 1: Old architecture (64, 32) + relu
    print("\n  [Test 1] Old architecture: (64,32) + relu")
    scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                            max_iter=1000, random_state=42)
        mlp.fit(X_train_s, y_train)
        scores.append(accuracy_score(y_test, mlp.predict(X_test_s)))
    print(f"    Accuracy: {np.mean(scores):.4f}")

    # Test 2: New architecture (n, n) + tanh + lbfgs
    print("\n  [Test 2] New architecture: (n,n) + tanh + lbfgs")
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        mlp = MLPClassifier(hidden_layer_sizes=(n_bits, n_bits),
                            activation='tanh', solver='lbfgs',
                            max_iter=2000, random_state=42)
        mlp.fit(X_train_s, y_train)
        scores.append(accuracy_score(y_test, mlp.predict(X_test_s)))
    print(f"    Accuracy: {np.mean(scores):.4f}")

    # Test 3: No scaling, small network
    print("\n  [Test 3] No scaling + (n,n) + tanh + lbfgs")
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        mlp = MLPClassifier(hidden_layer_sizes=(n_bits, n_bits),
                            activation='tanh', solver='lbfgs',
                            max_iter=2000, random_state=42)
        mlp.fit(X_train, y_train)
        scores.append(accuracy_score(y_test, mlp.predict(X_test)))
    print(f"    Accuracy: {np.mean(scores):.4f}")