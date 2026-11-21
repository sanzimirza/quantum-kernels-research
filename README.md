# A Hybrid Classical-Quantum Ensemble for Parity Classification: Quantum Kernels Dominate in the Small-Data Regime

## 📝 Abstract
Abstract—Parity problems (XOR function) can be solved using MLPs with a SVM and RBF kernel but there is still one problem. Their performance collapses as the parity problem scales to higher dimensions like 11 bit parity classification. This scalability bottleneck has been addressed by researchers working in quantum machine learning by using quantum kernels. They outperform classical RBF kernel by over 12% on complex parity-structured data. However there are still constraints we have to work around. Pure quantum models have to deal with limited qubit coherence, hardware noise and practical scalability challenges that makes them unreliable as singular standalone solutions. To counter the unreliability of pure quantum model, in this paper we will try to combine the strengths of both models with working around their respective weaknesses. Hence, we propose a ”Hybrid Classical Quantum Ensemble (HCQE)”. The architecture combines a classical MLP with a Quantum Support Vector Classifier (QSVC). We hypothesize that the accuracy of HCQE will land between the QSVC and the standalone classical model. This might make it more reliable than quantum models. Index Terms—parity problems, HCQE, SVM, RBF, Quantum Kernels, MLPs, QSVC

## 🔬 My Contribution
 I am the first author of this paper. I implemented the quantum kernel estimation circuits, analyzed the feature map performance and got the benchmark results.

## 📂 Repository Contents
* **Paper:** The full PDF is available in this repository.
* **Code:** The Python scripts/notebooks used to build the quantum feature maps and run the kernel methods.
* **Results:** The plots, graphs, and datasets from the paper.

## ⚙️ How to Use
1. Read the paper (PDF provided).
2. Install required packages: `pip install numpy matplotlib qiskit scikit-learn` (adjust as needed).
3. Open the code files to view the quantum kernel implementation and reproduce the results.

## 📫 Contact
**Sanjari Mirza** 
sanjarimirza6@gmail.com
