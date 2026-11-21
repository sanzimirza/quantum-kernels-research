"""
Hardware verification: 3-bit parity QSVC on real IBM Quantum hardware.
Uses minimal samples + shots to fit within free tier.
"""
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


def generate_parity_data(n_bits):
    X = np.array(list(product([0, 1], repeat=n_bits)))
    y = np.array([sum(row) % 2 for row in X])
    return X, y


from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def build_hardware_qsvc(n_bits):
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Using backend: {backend.name}")

    # Create pass manager for this specific backend
    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend
    )

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 100

    # Pass the pass_manager here - this transpiles circuits automatically
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pass_manager)
    feature_map = zz_feature_map(feature_dimension=n_bits, reps=1)

    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    return QSVC(quantum_kernel=kernel), backend.name


if __name__ == "__main__":
    # 3-bit parity = 8 samples total
    X, y = generate_parity_data(n_bits=3)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} bits")

    # Use just 6 samples for training, 2 for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=2, random_state=42, stratify=y
    )

    print(f"Training QSVC on real hardware with {len(X_train)} samples...")
    qsvc, backend_name = build_hardware_qsvc(n_bits=3)
    qsvc.fit(X_train, y_train)

    y_pred = qsvc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  HARDWARE VERIFICATION RESULT")
    print(f"{'='*50}")
    print(f"  Backend: {backend_name}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Predictions:   {y_pred}")
    print(f"  Ground truth:  {y_test}")