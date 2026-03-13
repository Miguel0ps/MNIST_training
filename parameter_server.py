import socket
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

HOST = "10.253.37.236"
PORT = 5000
NUM_WORKERS = 1
EPOCHS = 100  # Por ahora pequeño para pruebas rápidas
LR = 0.1     # Learning rate locked at 0.1

# Cargar datos de prueba para evaluar el accuracy global
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(-1, 784) / 255.0

def initialize_parameters(input_size, hidden_size, output_size):
    # np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def compute_accuracy(X, y_true, W1, b1, W2, b2):
    # Forward pass simple para evaluación
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    predictions = np.argmax(Z2, axis=1)
    return np.mean(predictions == y_true)

def send_data(conn, data):
    serialized = pickle.dumps(data)
    conn.sendall(len(serialized).to_bytes(8, 'big'))
    conn.sendall(serialized)

def recv_data(conn):
    try:
        size_data = conn.recv(8)
        if not size_data: return None
        size = int.from_bytes(size_data, 'big')
        data = b''
        while len(data) < size:
            packet = conn.recv(4096)
            if not packet: break
            data += packet
        return pickle.loads(data)
    except:
        return None

# Inicializar modelo y métricas
W1, b1, W2, b2 = initialize_parameters(784, 128, 10)
history_acc = []

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print(f"Servidor iniciado. Esperando {NUM_WORKERS} workers...")

workers = []
for i in range(NUM_WORKERS):
    conn, addr = server.accept()
    print(f"Worker {i + 1} conectado desde {addr}")
    # ASIGNACIÓN AUTOMÁTICA DE ID: Enviamos el ID y el total de workers al inicio
    send_data(conn, {"worker_id": i, "total_workers": NUM_WORKERS})
    workers.append(conn)

# Bucle de entrenamiento
for epoch in range(EPOCHS):
    # 1. Enviar pesos a todos
    for conn in workers:
        send_data(conn, (W1, b1, W2, b2))

    # 2. Recibir gradientes
    grads = []
    for conn in workers:
        grad = recv_data(conn)
        if grad: grads.append(grad)

    # 3. Promediar y Actualizar
    if len(grads) == NUM_WORKERS:
        dW1 = sum(g[0] for g in grads) / NUM_WORKERS
        db1 = sum(g[1] for g in grads) / NUM_WORKERS
        dW2 = sum(g[2] for g in grads) / NUM_WORKERS
        db2 = sum(g[3] for g in grads) / NUM_WORKERS

        W1 -= LR * dW1
        b1 -= LR * db1
        W2 -= LR * dW2
        b2 -= LR * db2

    # 4. Calcular Accuracy al final de la época
    acc = compute_accuracy(X_test, y_test, W1, b1, W2, b2)
    history_acc.append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} - Accuracy: {acc:.4f}")

print("Entrenamiento finalizado. Cerrando conexiones...")
for conn in workers: conn.close()
server.close()

# Graficar resultados
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), history_acc, marker='o', color='b', linestyle='-')
plt.title('Precisión del Modelo (MNIST) - Entrenamiento Distribuido')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()