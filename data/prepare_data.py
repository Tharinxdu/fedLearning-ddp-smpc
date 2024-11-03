import time
import os
import json
import tensorflow as tf
import numpy as np

start_time = time.time()

def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    return (train_images, train_labels), (test_images, test_labels)

def partition_data(num_clients=10):
    (train_images, train_labels), _ = load_mnist()
    client_data = {}
    data_per_client = len(train_images) // num_clients
    for i in range(num_clients):
        start = i * data_per_client
        end = start + data_per_client
        client_data[f'client_{i+1}'] = {
            'images': train_images[start:end].tolist(),
            'labels': train_labels[start:end].tolist()
        }
    os.makedirs('data/partitions/', exist_ok=True)
    with open('data/partitions/client_data.json', 'w') as f:
        json.dump(client_data, f)

if __name__ == "__main__":
    partition_data(num_clients=10)
    print(f"Partitioning completed in {time.time() - start_time:.2f} seconds.")
