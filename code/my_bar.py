import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
import numpy as np
import pickle

exp_freq = 13.353341852585672
sim_freq = 64
desired_freq = 10

def load_experimental_data():
    # Load the CSV file into a DataFrame
    file_path = '~/Documents/university/MRIGI/group-project-70006/real-world-experiment-data/my-merged-3.csv'
    data = pd.read_csv(file_path)

    # Extract the electrode columns and coordinate columns
    num_electrode_columns = 10
    num_coord_columns = 2

    # Assuming the format is: time, electrodes, coordinates
    electrode_columns = data.iloc[:, 1:1 + num_electrode_columns]
    coord_columns = data.iloc[:, 1 + num_electrode_columns:1 + num_electrode_columns + num_coord_columns]

    # Convert to numpy arrays
    x_numpy = electrode_columns.to_numpy()  # num_samples x num_electrodes
    y_numpy = coord_columns.to_numpy()      # num_samples x num_coords

    # Print shapes for confirmation
    print("x_numpy shape:", x_numpy.shape)
    print("y_numpy shape:", y_numpy.shape)

    return x_numpy, y_numpy

def load_simulation_data():
    input_file_path = "/Users/piotrblaszyk/Documents/university/MRIGI/group-project-70006/EIDORS_FEM/simulation_results.pkl"
    with open(input_file_path, "rb") as file:
        foo = pickle.load(file, encoding='latin1')
    x = np.array(foo[0])
    y = np.array(foo[1])
    return x, y

def plot_data(y_test_numpy, y_pred_numpy):
    # Assume y_test_numpy and y_pred_numpy are num_samples x n arrays.
    # y_test_numpy: Ground truth values (num_samples x n)
    # y_pred_numpy: Predicted values (num_samples x n)

    n = y_test_numpy.shape[1]  # number of dimensions

    # Plot the true vs predicted values for each dimension
    for i in range(n):
        plt.figure(figsize=(8, 6))
        
        # Extract specific dimension (i)
        true_dim = y_test_numpy[:, i]
        pred_dim = y_pred_numpy[:, i]
        
        # Plot true vs predicted for the i-th dimension
        plt.scatter(true_dim, pred_dim, label=f"Predicted vs True (Dim {i+1})", color="red", alpha=0.6)
        
        # Plot a line y=x (45-degree line) for reference
        min_val = min(np.min(true_dim), np.min(pred_dim))
        max_val = max(np.max(true_dim), np.max(pred_dim))
        foo = np.linspace(min_val, max_val, 500)  # Adjust the range as needed
        plt.plot(foo, foo, 'r--')  # 'r--' specifies a red dotted line
        
        plt.xlabel(f"True Dim {i+1}")
        plt.ylabel(f"Pred Dim {i+1}")
        plt.title(f"True vs Predicted Values: Dim {i+1}")
        plt.grid(True)
        plt.legend()
        
        # Show the plot for each dimension
        plt.tight_layout()
        plt.show()

    line_colors = ['green']

    # Create scatter plots for each pair of dimensions (dim1 vs dim2, dim1 vs dim3, etc.)
    for i in range(n):
        for j in range(i+1, n):
            plt.figure(figsize=(8, 6))
            
            true_dim_i = y_test_numpy[:, i]
            true_dim_j = y_test_numpy[:, j]
            pred_dim_i = y_pred_numpy[:, i]
            pred_dim_j = y_pred_numpy[:, j]

            # Plot the scatter for the i-th and j-th dimension
            plt.scatter(true_dim_i, true_dim_j, label="True", color="blue", alpha=0.6)
            plt.scatter(pred_dim_i, pred_dim_j, label="Predicted", color="red", alpha=0.6)
            
            # Plot a green line between corresponding true and predicted points
            if True:
                for k in range(len(true_dim_i)):
                    color = line_colors[k % len(line_colors)]
                    plt.plot([true_dim_i[k], pred_dim_i[k]], [true_dim_j[k], pred_dim_j[k]], 
                            color=color, linestyle="-", linewidth=0.5)

            plt.xlabel(f"Dimension {i+1}")
            plt.ylabel(f"Dimension {j+1}")
            plt.title(f"True vs Predicted Values: (Dim {i+1}, Dim {j+1})")
            plt.legend()
            plt.grid(True)

            # Show the plot for each pair of dimensions
            plt.tight_layout()
            plt.show()
    
    y_test = y_test_numpy
    y_pred = y_pred_numpy
    x_axis = list(range(y_test.shape[0]))
    
    for i in range(y_test.shape[1]):
        plt.figure(figsize=(8, 5))  # Create a new figure for each dimension
        plt.plot(x_axis, y_test[:, i], label=f"y_test Dimension {i+1}", color="blue", linestyle="--")
        plt.plot(x_axis, y_pred[:, i], label=f"y_pred Dimension {i+1}", color="red", linestyle="-")
        plt.xlabel("Time [au]")
        plt.ylabel("Value [mm]")
        plt.title(f"Comparison for Output Dimension {i+1}")
        plt.legend()
        plt.grid(True)
        plt.show()

def _whammer(file_path, variable_name):
    data = loadmat(file_path)
    return torch.tensor(data[variable_name], dtype=torch.float32)

def load_sciro_data():
    path = "/Users/piotrblaszyk/Documents/university/MRIGI/group-project-70006/SciRo/"
    x_train = _whammer(f"{path}x_train.mat", "XTrain").T
    y_train = _whammer(f"{path}y_train.mat", "YTrain").T
    x_test = _whammer(f"{path}x_test.mat", "XTest").T
    y_test = _whammer(f"{path}y_test.mat", "YTest").T

    x = torch.cat((x_train, x_test), dim=0).numpy()
    y = torch.cat((y_train, y_test), dim=0).numpy()

    return x, y

def test_model(model, test_loader):
    # Evaluate the model and collect predictions
    model.eval()  # Set the model to evaluation mode
    y_pred_list = []
    y_test_list = []

    with torch.no_grad():  # Disable gradient computation for inference
        for x_batch, y_batch in test_loader:
            y_pred_batch = model(x_batch)  # Predict
            y_pred_list.append(y_pred_batch)
            y_test_list.append(y_batch)

    # Concatenate predictions and true values into single tensors
    y_pred = torch.cat(y_pred_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)

    # Convert to NumPy arrays if needed
    y_pred_numpy = y_pred.numpy()
    y_test_numpy = y_test.numpy()

    # Print shapes for confirmation
    print("y_pred shape:", y_pred.shape)
    print("y_test shape:", y_test.shape)

    return y_test_numpy, y_pred_numpy

def print_mse(y_test, y_pred):
    # Assuming y_pred and y_test are PyTorch tensors
    mse = F.mse_loss(torch.from_numpy(y_pred), torch.from_numpy(y_test))
    my_test_loss_2 = mse.item()

    print(f"Mean Squared Error (PyTorch): {my_test_loss_2:.0e}")
    print(f"Mean Squared Error (PyTorch): {my_test_loss_2:.4e}")

# Normalize function
def normalize(data):
    means = np.mean(data, axis=0)  # Mean of each column
    stds = np.std(data, axis=0)    # Standard deviation of each column
    normalized_data = (data - means) / stds  # Normalize
    return normalized_data, means, stds

# De-normalize function
def denormalize(normalized_data, means, stds):
    return (normalized_data * stds) + means  # De-normalize

def chunk_sequence(x_numpy, y_numpy, sequence_length):
    num_samples = x_numpy.shape[0]
    num_input_dims = x_numpy.shape[1]
    num_output_dims = y_numpy.shape[1]

    # Initialize new arrays
    x = []
    y = []

    # Create sequences
    for ix in range(sequence_length - 1, num_samples):
        x_sequence = x_numpy[ix - sequence_length + 1 : ix + 1]  # Extract sequence of length `sequence_length`
        x.append(x_sequence)
        y.append(y_numpy[ix])  # Match corresponding y_numpy

    # Convert to numpy arrays
    x = np.array(x)  # Shape: (num_samples - sequence_length + 1, sequence_length, num_input_dims)
    y = np.array(y)  # Shape: (num_samples - sequence_length + 1, num_output_dims)

    return x, y

def prepare_dataloader(x, y, batch_size, shuffle):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, persistent_workers=True)

def train_validation_test(x, y):
    num_samples = x.shape[0]
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)

    indices = list(range(num_samples))
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    x_train, y_train = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]
    x_test, y_test = x[test_indices], y[test_indices]

    return x_train, y_train, x_val, y_val, x_test, y_test

def train_validation(x, y):
    num_samples = x.shape[0]
    train_end = int(0.8 * num_samples)

    indices = list(range(num_samples))
    train_indices = indices[:train_end]
    val_indices = indices[train_end:]

    x_train, y_train = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]

    return x_train, y_train, x_val, y_val

def resample(x, y, ratio):
    num_samples = x.shape[0]
    z = np.hstack((x, y))

    z_df = pd.DataFrame(z)
    z_df.insert(0, 'Index', np.arange(num_samples) / (num_samples - 1))
    desired_num_samples = int(ratio * num_samples)
    foo = pd.DataFrame({'Index': np.linspace(0, 1, desired_num_samples)})
    bar = pd.merge(foo, z_df, on='Index', how='outer')
    bar = bar.sort_values(by='Index')
    bar_interpolated = bar.interpolate(method='linear')
    bar_cleaned = bar_interpolated.dropna()
    filtered_bar = bar_cleaned[bar_cleaned['Index'].isin(foo['Index'])]
    bar_reset = filtered_bar.reset_index(drop=True)
    bar_no_index = bar_reset.drop('Index', axis=1)
    bar_numpy = bar_no_index.to_numpy()

    x_resampled = bar_numpy[:, 0 : x.shape[1]]
    y_resampled = bar_numpy[:, x.shape[1] : ]
    return x_resampled, y_resampled

def intelligent_resample(x_sim, y_sim, x_exp, y_exp):
    speeds = np.zeros(2)
    ys = [y_sim, y_exp]
    for i in range(2):
        y = ys[i]
        differences = np.diff(y, axis=0)
        norms = np.linalg.norm(differences, axis=1)
        speeds[i] = np.mean(norms)
    ratio = speeds[0] / speeds[1]
    # if ratio = 2, then each sim step involves larger movement than exp step
    # so need to upsample sim
    # if ratio > 1, upsample sim
    # if ratio < 1, downsample sim
    x_sim_resampled, y_sim_resampled = resample(x_sim, y_sim, ratio)
    return x_sim_resampled, y_sim_resampled, x_exp, y_exp
