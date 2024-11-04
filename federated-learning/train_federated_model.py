import json
import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import os
import time
import numpy as np
from models.base_model import create_cnn_model


# Function to load client data
def load_client_data():
    with open('../data/data/partitions/client_data.json', 'r') as f:
        client_data = json.load(f)
    federated_data = {}
    for client_id, data in client_data.items():
        images = tf.data.Dataset.from_tensor_slices(data['images'])
        labels = tf.data.Dataset.from_tensor_slices(data['labels'])
        federated_data[client_id] = tf.data.Dataset.zip((images, labels)).batch(20)
    return federated_data


# Load federated data
federated_data = list(load_client_data().values())


# Load MNIST test data for evaluation
def load_test_data():
    (test_images, test_labels), _ = tf.keras.datasets.mnist.load_data()
    test_images = test_images.reshape(-1, 28, 28, 1) / 255.0
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    return test_data.batch(20)


test_data = load_test_data()


# Function to calculate model size in bytes
def calculate_model_size(model):
    total_size = 0
    for weight in model.weights:
        total_size += np.prod(weight.shape) * weight.dtype.size
    return total_size


# Define the model for TFF
def model_fn():
    keras_model = create_cnn_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )


# Initialize federated learning process
client_optimizer_fn = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
server_optimizer_fn = tff.learning.optimizers.build_sgdm(learning_rate=1.0)

# Initialize federated learning process
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=client_optimizer_fn,  # Corrected optimizer type
    server_optimizer_fn=server_optimizer_fn   # Corrected optimizer type
)
state = iterative_process.initialize()

# Set up logging folder
log_folder = '../logs'  # Use the existing logs folder at the specified path
log_file_path = os.path.join(log_folder, 'FL_Base_MNIST_log.csv')

# Federated training loop
NUM_ROUNDS = 10
results = []
round_times = []
communication_overheads = []

initial_model = create_cnn_model()
initial_model_size = calculate_model_size(initial_model)

for round_num in range(1, NUM_ROUNDS + 1):
    start_time = time.time()  # Start timer for the round
    state, metrics = iterative_process.next(state, federated_data)

    # Evaluate on test data after each round
    keras_model = create_cnn_model()
    state.model.assign_weights_to(keras_model)
    test_loss, test_accuracy = keras_model.evaluate(test_data, verbose=0)

    # Calculate communication overhead
    updated_model_size = calculate_model_size(keras_model)
    round_communication = updated_model_size * len(federated_data)  # Simulate transmission to server and back
    communication_overheads.append(round_communication)

    # End timer for the round
    round_duration = time.time() - start_time
    round_times.append(round_duration)

    # Log metrics
    print(f'Round {round_num}, Train Metrics: {metrics}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, '
          f'Round Duration: {round_duration:.2f} seconds, Communication Overhead: {round_communication / 1e6:.2f} MB')
    results.append({
        'Round': round_num,
        'Train_Loss': metrics['train']['loss'],
        'Train_Accuracy': metrics['train']['sparse_categorical_accuracy'],
        'Test_Loss': test_loss,
        'Test_Accuracy': test_accuracy,
        'Round_Duration_Seconds': round_duration,
        'Communication_Overhead_MB': round_communication / 1e6  # Convert to MB for readability
    })

# Calculate total training time and average round duration
total_training_time = sum(round_times)
average_round_duration = total_training_time / NUM_ROUNDS

# Append the total training time and average round duration to the results
results.append({
    'Round': 'Total',
    'Train_Loss': '',
    'Train_Accuracy': '',
    'Test_Loss': '',
    'Test_Accuracy': '',
    'Round_Duration_Seconds': total_training_time,
    'Communication_Overhead_MB': sum(communication_overheads) / 1e6  # Total communication in MB
})
results.append({
    'Round': 'Average',
    'Train_Loss': '',
    'Train_Accuracy': '',
    'Test_Loss': '',
    'Test_Accuracy': '',
    'Round_Duration_Seconds': average_round_duration,
    'Communication_Overhead_MB': (sum(communication_overheads) / NUM_ROUNDS) / 1e6  # Average communication in MB
})

# Save the results to CSV
df = pd.DataFrame(results)
df.to_csv(log_file_path, index=False)
print(f'Results saved to {log_file_path}')
