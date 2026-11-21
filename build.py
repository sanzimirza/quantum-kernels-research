from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute

def build_qsvc(n_bits):
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False)

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        fidelity = ComputeUncompute(sampler=sampler)
        feature_map = zz_feature_map(feature_dimension=n_bits, reps=2)
        kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
        return QSVC(quantum_kernel=kernel)