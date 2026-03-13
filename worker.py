import socket
import pickle
import numpy as np
from tensorflow.keras.datasets import mnist

HOST = "10.253.46.239"
PORT = 5000

def one_hot(y, num_classes=10):
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded

def relu(Z): return np.maximum(0, Z)
def relu_derivative(Z): return Z > 0
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward(X, Y, W1, b1, W2, b2):
    m = X.shape[0]
    Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
    dZ2 = A2 - Y
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def send_data(sock, data):
    serialized = pickle.dumps(data)
    sock.sendall(len(serialized).to_bytes(8, 'big'))
    sock.sendall(serialized)

def recv_data(sock):
    size_data = sock.recv(8)
    if not size_data: return None
    size = int.from_bytes(size_data, 'big')
    data = b''
    while len(data) < size:
        packet = sock.recv(4096)
        if not packet: break
        data += packet
    return pickle.loads(data)

# Conexión inicial
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

# RECIBIR CONFIGURACIÓN AUTOMÁTICA
config = recv_data(sock)
WORKER_ID = config['worker_id']
NUM_WORKERS = config['total_workers']
print(f"Soy el Worker {WORKER_ID + 1} de {NUM_WORKERS}")

# Cargar y partir dataset
(X_train, y_train), _ = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
y_train = one_hot(y_train)

X_split = np.array_split(X_train, NUM_WORKERS)[WORKER_ID]
Y_split = np.array_split(y_train, NUM_WORKERS)[WORKER_ID]

while True:
    data = recv_data(sock)
    if data is None: break # El servidor cerró la conexión
    
    W1, b1, W2, b2 = data
    grads = backward(X_split, Y_split, W1, b1, W2, b2)
    send_data(sock, grads)

sock.close()