from config import get_dl_config
import xarray as xr
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time
import os 
import json
from tqdm import tqdm

conf = get_dl_config()
SAMPLING_FREQ = conf['sampling_frequency']
INPUT_FILE = conf['input_file']
OUTPUT_PATH = "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/results/"

BATCH_SIZE = 128 #128
LR = 1e-3 

N_EPOCHS = 100
N_SPLITS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

class TUHDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0), self.labels[idx]

class SimpleNet(nn.Module):
    def __init__(self, n_samples=5, domain='time'):
        super(SimpleNet, self).__init__()
        # input: (batch_size, 1, 19, n_samples )

        self.conv = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=1)         
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2) 

        self.fc_input_size = 6 * 7 * ((n_samples - 4) // 2)
        self.fc = nn.Linear(self.fc_input_size, 1)  


    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)  
        x = self.fc(x)

        x = torch.sigmoid(x)  

        return x

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    start_time = time.time()  

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct_predictions += ((outputs.squeeze() > 0.5) == targets).sum().item()
        total_samples += targets.size(0)

    epoch_time = time.time() - start_time 

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy, epoch_time

def validate_model(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    start_time = time.time()  

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            total_loss += loss.item()

            predictions = (outputs.squeeze() > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            correct_predictions += ((predictions == targets).sum().item())
            total_samples += targets.size(0)
        
    val_time = time.time() - start_time 

    loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    roc_auc = roc_auc_score(all_targets, all_predictions)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()

    return (
        float(f"{loss:.4f}"),
        float(f"{accuracy:.4f}"),
        float(f"{precision:.4f}"),
        float(f"{recall:.4f}"),
        float(f"{f1:.4f}"),
        float(f"{roc_auc:.4f}"),
        int(tn), int(fp), int(fn), int(tp),
        float(f"{val_time:.4f}"),
    )

def cross_validate_model(n_samples, measurements, labels, patient_ids, n_splits, batch_size, n_epochs, learning_rate, device):
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    splits = list(sgkf.split(X=measurements, y=labels, groups=patient_ids))

    split_metrics = []

    # Iterate over the splits
    for i, (train_idx, val_idx) in enumerate(splits):
        model = SimpleNet(n_samples=n_samples).to(device)

        # Prepare the data
        train_data, train_labels = measurements[train_idx], labels[train_idx]
        val_data, val_labels = measurements[val_idx], labels[val_idx]

        train_dataset = TUHDataset(train_data, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TUHDataset(val_data, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize optimizer and loss function
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_epoch_metrics = {
            'epoch': 0,
            'val_loss': np.inf,
            'val_accuracy': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0,
            'val_f1': 0.0,
            'val_roc_auc': 0.0,
            'val_tn': 0,
            'val_fp': 0,
            'val_fn': 0,
            'val_tp': 0,
            'total_train_time': 0.0,
            'val_time': 0.0, 
        }

        total_train_time = 0.0
        
        # Training loop with tqdm progress bar
        for epoch in tqdm(range(n_epochs), desc=f"Split {i + 1}/{n_splits}"):
            train_loss, train_accuracy, train_time = train_epoch(model, train_dataloader, optimizer, criterion, device)
            total_train_time += train_time
            total_train_time = float(f"{total_train_time:.4f}")

            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, tn, fp, fn, tp, val_time = validate_model(model, val_dataloader, criterion, device)
        
            if val_loss < best_epoch_metrics['val_loss']:
                best_epoch_metrics = {
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1,
                    'val_roc_auc': val_roc_auc, 
                    "val_tn": int(tn),
                    "val_fp": int(fp),
                    "val_fn": int(fn),
                    "val_tp": int(tp),
                    'total_train_time': total_train_time,
                    'val_time': val_time,
                }

                tqdm.write(f"New best Epoch {epoch + 1}/{n_splits}, "
                           f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            else:
                tqdm.write(f'Epoch {epoch + 1}/{n_epochs}, '
                           f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                           f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        split_metrics.append(best_epoch_metrics)

        print(f"Split {i + 1}/{n_splits}, best epoch at {best_epoch_metrics['epoch']}, "
              f"Val Loss: {best_epoch_metrics['val_loss']:.4f}, "
              f"Val Accuracy: {best_epoch_metrics['val_accuracy']:.4f}, ")
            
        # Explicitly delete the data and model to free up memory
        del train_data, train_labels, val_data, val_labels
        del train_dataset, val_dataset, train_dataloader, val_dataloader
        del model, optimizer
        torch.cuda.empty_cache()  # Clear the GPU cache
    
    avg_metrics = {
        'avg_epoch': float(np.mean([m['epoch'] for m in split_metrics])),
        'avg_val_loss': float(np.mean([m['val_loss'] for m in split_metrics])),
        'avg_val_accuracy': float(np.mean([m['val_accuracy'] for m in split_metrics])),
        'avg_val_precision': float(np.mean([m['val_precision'] for m in split_metrics])),
        'avg_val_recall': float(np.mean([m['val_recall'] for m in split_metrics])),
        'avg_val_f1': float(np.mean([m['val_f1'] for m in split_metrics])),
        'avg_val_roc_auc': float(np.mean([m['val_roc_auc'] for m in split_metrics])),
        'sum_val_tn': int(np.sum([m['val_tn'] for m in split_metrics])),
        'sum_val_fp': int(np.sum([m['val_fp'] for m in split_metrics])),
        'sum_val_fn': int(np.sum([m['val_fn'] for m in split_metrics])),
        'sum_val_tp': int(np.sum([m['val_tp'] for m in split_metrics])),
        'avg_total_train_time': float(np.mean([m['total_train_time'] for m in split_metrics])),
        'avg_val_time': float(np.mean([m['val_time'] for m in split_metrics])),
    }

    return split_metrics, avg_metrics

input_files = [
    #"/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_5.nc",
    #"/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_10.nc",
    "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_15.nc",
    #"/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_20.nc",
    #"/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_25.nc",
    #"/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_30.nc",
]

for input_file in input_files:
    data = xr.open_dataarray(input_file)       
    print(f"Loading data from {input_file} with shape {data.shape}...")

    labels = data['label'].values
    patient_ids = data['patient_id'].values
    measurements = data.values

    print(f"Loaded data.")

    n_samples = data.sizes["time"] 

    # determine window_length based domain of data
    file_name = os.path.basename(input_file)
    domain = None
    if 'time' in file_name:
        window_length = n_samples // SAMPLING_FREQ
        domain = 'time'
    elif'freq' in file_name:
        window_length = n_samples * 2 # took only positive fft values
        domain = 'freq'
    else:
        raise ValueError(f"Unknown domain for file {file_name}")

    print(f"Loaded data with shape {data.shape} in {domain} domain")

    window_length = n_samples // SAMPLING_FREQ

    print(f"Training CNN with window length {window_length}")

    split_metrics, avg_metrics = cross_validate_model(n_samples, measurements, labels, patient_ids, n_splits=N_SPLITS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, learning_rate=LR, device=device)

    print(f"Finished training, avg metrics: {avg_metrics}")

    results = {
        "splits": split_metrics,
        "avg": avg_metrics
    }

    print(f"Saving results...")
    results_output = os.path.join(OUTPUT_PATH, f"cnn_{window_length}_{BATCH_SIZE}_{LR}.json")
    with open(results_output, 'w') as f:
        json.dump(results, f)

    del data, labels, patient_ids, measurements

    print(f"Results saved to {results_output}.")

