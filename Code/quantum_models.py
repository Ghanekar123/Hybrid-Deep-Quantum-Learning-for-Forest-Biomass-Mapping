# =============================================================================
# Step 6 — Quantum Regression
# =============================================================================
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVR
from config import N_QUBITS,N_LAYERS,SEED
from transformer import compute_metrics

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    print("[warn] PennyLane not installed. Quantum steps will be skipped.")

if HAS_PENNYLANE:
    QDEV = qml.device("default.qubit",wires=N_QUBITS)

    def _encode_features(x,n_qubits=N_QUBITS):
        x = np.asarray(x,dtype=np.float64)
        if len(x)<n_qubits:
            x = np.pad(x,(0,n_qubits-len(x)))
        return np.tanh(x[:n_qubits])*np.pi

    @qml.qnode(QDEV)
    def _kernel_circuit(x1,x2):
        a = _encode_features(x1)
        b = _encode_features(x2)
        for _ in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.Hadamard(wires=i)
                qml.RZ(a[i],wires=i)
                qml.RY(b[i],wires=i)
            for i in range(N_QUBITS-1):
                qml.CNOT(wires=[i,i+1])
        return qml.probs(wires=range(N_QUBITS))

    def quantum_kernel(X1,X2,verbose=False):
        K = np.zeros((len(X1),len(X2)))
        t0 = time.time()
        for i,x1 in enumerate(X1):
            for j,x2 in enumerate(X2):
                K[i,j] = _kernel_circuit(x1,x2)[0]
            if verbose and (i+1)%max(1,len(X1)//5)==0:
                print(f"[qk] {i+1}/{len(X1)} rows done ({time.time()-t0:.1f}s)")
        return K

    class QSVR:
        def __init__(self,C=10.0,epsilon=0.1):
            self.C = C
            self.epsilon = epsilon
        def fit(self,X,y):
            self.X_train_ = np.asarray(X)
            self.y_train_ = np.asarray(y)
            K = quantum_kernel(self.X_train_,self.X_train_,verbose=True)
            self.svr_ = SVR(kernel="precomputed",C=self.C,epsilon=self.epsilon)
            self.svr_.fit(K,self.y_train_)
            return self
        def predict(self,X):
            K = quantum_kernel(np.asarray(X),self.X_train_,verbose=True)
            return self.svr_.predict(K)

    class VQR:
        def __init__(self,in_dim,n_qubits=N_QUBITS,n_layers=N_LAYERS,lr=5e-3,epochs=200,batch_size=16,seed=SEED):
            torch.manual_seed(seed)
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.epochs = epochs
            self.batch_size = batch_size
            self.proj = nn.Linear(in_dim,n_qubits)
            self.q_weights = nn.Parameter(0.1*torch.randn(n_layers,n_qubits,2))
            self.params = list(self.proj.parameters())+[self.q_weights]
            self.opt = torch.optim.Adam(self.params,lr=lr)
            self._build_qnode()
        def _build_qnode(self):
            n_q,n_l = self.n_qubits,self.n_layers
            @qml.qnode(QDEV,interface="torch",diff_method="backprop")
            def circuit(inputs,weights):
                for l in range(n_l):
                    for i in range(n_q):
                        qml.RY(inputs[i],wires=i)
                        qml.RZ(weights[l,i,0],wires=i)
                        qml.RY(weights[l,i,1],wires=i)
                    for i in range(n_q-1):
                        qml.CNOT(wires=[i,i+1])
                return qml.expval(qml.PauliZ(0))
            self.circuit = circuit
        def _forward(self,X):
            angles = torch.tanh(self.proj(X))*np.pi
            outs = [self.circuit(angles[b],self.q_weights) for b in range(angles.shape[0])]
            return torch.stack(outs)
        def fit(self,X,y):
            X = torch.tensor(X,dtype=torch.float32)
            y = torch.tensor(y,dtype=torch.float32)
            n = X.shape[0]
            for epoch in range(self.epochs):
                perm = torch.randperm(n)
                for i in range(0,n,self.batch_size):
                    idx = perm[i:i+self.batch_size]
                    self.opt.zero_grad()
                    pred = self._forward(X[idx])
                    loss = torch.mean((pred-y[idx])**2)
                    loss.backward()
                    self.opt.step()
            return self
        def predict(self,X):
            X = torch.tensor(X,dtype=torch.float32)
            with torch.no_grad():
                return self._forward(X).numpy()
else:
    QSVR = None
    VQR = None

def run_quantum_regressors(data,Z,C=10.0,epsilon=0.1,vqr_epochs=150,vqr_lr=5e-3):
    print("\n"+"="*70)
    print("STEP 6: Quantum Regression (QSVR & VQR)")
    print("="*70)
    if not HAS_PENNYLANE:
        print("[skip] PennyLane unavailable.")
        return {},{}
    Ztr,Zte = Z["Z_train"],Z["Z_test"]
    ytr = data["y_train"]
    results,models = {},{}
    print("\n[qsvr] Training QSVR...")
    t0 = time.time()
    try:
        qsvr = QSVR(C=C,epsilon=epsilon).fit(Ztr,ytr)
        pred_s = qsvr.predict(Zte)
        pred = pred_s*data["y_std"]+data["y_mean"]
        m = compute_metrics(data["y_test_raw"],pred)
        m["train_time_s"] = round(time.time()-t0,2)
        m["n_qubits"] = N_QUBITS
        results["QSVR"],models["QSVR"] = m,qsvr
    except Exception as e:
        print(f"[qsvr] Failed: {e}")
    print("\n[vqr] Training VQR...")
    t0 = time.time()
    try:
        vqr = VQR(Ztr.shape[1],N_QUBITS,N_LAYERS,vqr_lr,vqr_epochs,16)
        vqr.fit(Ztr,ytr)
        pred_s = vqr.predict(Zte)
        pred = pred_s*data["y_std"]+data["y_mean"]
        m = compute_metrics(data["y_test_raw"],pred)
        m["train_time_s"] = round(time.time()-t0,2)
        m["n_qubits"] = N_QUBITS
        results["VQR"],models["VQR"] = m,vqr
    except Exception as e:
        print(f"[vqr] Failed: {e}")
    return results,models