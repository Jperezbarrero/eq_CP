import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset, DataLoader
import glob 
import tables

# Parámetros
path = r"C:\Users\usuario\Desktop\UNI\5º\TFGFis\Programas\eq_CP\ttbar_delphes_forTraining\tag_2_delphes_events179.h5"
batch_size = 1000
epochs = 100
n_qubits = 4  # 2^4 = 16, suficiente para 14 variables + padding
n_layers = 3  # Capas en la red cuántica

# Dataset personalizado
class dataset(IterableDataset):
    def __init__(self, path):
        self.files = glob.glob(path)
        self.length = 0
        for fil in self.files:
            h5file = tables.open_file(fil, mode='r')
            self.length += h5file.root.df.axis1.shape[0]
            h5file.close()

    def __len__(self):
        return self.length

    def __iter__(self):
        for f in self.files:
            thedata = pd.read_hdf(f, 'df')
            control_vars = torch.Tensor(thedata[['parton_cnr_crn', 'parton_cnk_ckn', 'parton_crk_ckr']].values)
            weights = torch.Tensor(thedata[['weight_sm', 'weight_lin_ctGI', 'weight_quad_ctGI']].values)
            variables = torch.Tensor(thedata[['lep_pos_px', 'lep_pos_py', 'lep_pos_pz', 
                                              'lep_neg_px', 'lep_neg_py', 'lep_neg_pz', 
                                              'random_b1_px', 'random_b1_py', 'random_b1_pz', 
                                              'random_b2_px', 'random_b2_py', 'random_b2_pz',
                                              'met_px', 'met_py']].values)
            yield from zip(weights, control_vars, variables)

# Cargar los datos
training = dataset(path)
dataloader = DataLoader(training, batch_size)

# Definir dispositivo cuántico
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), normalize=True, pad_with=0)

def variational_circuit(weights):
    for i in range(n_layers):
        for j in range(n_qubits):
            qml.RY(weights[i, j], wires=j)  # Rotación en Y en cada qubit
        for k in range(n_qubits - 1):
            qml.CZ(wires=[k, k + 1])  # Conexiones entre qubits

@qml.qnode(dev, interface="torch")
def quantum_nn(inputs, weights):
    feature_map(inputs)
    variational_circuit(weights)
    return qml.expval(qml.PauliZ(0))  # Medición en el primer qubit

def loss_fn(input_vars, weights_qnn, dataset_weights):
    pred = quantum_nn(input_vars, weights_qnn)  # Ejecutar el circuito cuántico
    w_sm = dataset_weights[:, 0]  # Primera columna de pesos
    w_lin_ctGI = dataset_weights[:, 1]
    return torch.mean(w_sm * (pred - (w_lin_ctGI / w_sm)) ** 2)

# Inicializar los pesos del circuito cuántico
weights_qnn = torch.randn(n_layers, n_qubits, requires_grad=True)  # Estos son los parámetros entrenables

# Definir el optimizador
optimizer = torch.optim.Adam([weights_qnn], lr=0.1)

# Almacenar historial de pérdidas
train_loss_history = []

# Entrenamiento
for ep in range(epochs):
    print("Epoch", ep)
    loop = tqdm(dataloader)
    loss_per_batch = []

    for dataset_weights, control, input_vars in loop:
        optimizer.zero_grad()
        loss = loss_fn(input_vars, weights_qnn, dataset_weights)
        loss_per_batch.append(loss.item())
        loss.backward()
        optimizer.step()
    
    train_loss_history.append(sum(loss_per_batch) / len(loss_per_batch))
    print(f"Epoch {ep+1}/{epochs}, Loss: {train_loss_history[-1]:.5f}")

# Graficar la evolución de la pérdida
plt.plot(train_loss_history, label="Train Loss")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.show()

