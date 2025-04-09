from ZDRN01 data_preprocessing import preprocess_data, convert_to_balance_data, custom_sampling
from ZDRN01 model import ZDRN
from ZDRN01 training import train_model
from ZDRN01utils import setup_logging, setup_matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from sklearn.model_selection import KFold
import numpy as np

# 超参数配置
hyperparameters = {
    'lr': 0.001,
    'batch_size': 32,
    'weight_decay': 0.0001,
    'num_branches': 3,
    'num_layers': 2,
    'l1_lambda': 0.001,
    'total_iterations': 500,
    'patience': 20,
    'min_loss_improvement': 0.001
}


def main():
    setup_logging()
    setup_matplotlib()

    data_samples = convert_to_balance_data(jewelry_extended_info)
    X, y = preprocess_data(data_samples)
    X_resampled, y_resampled = custom_sampling(X, y)

    num_classes = len(np.unique(y_resampled))
    input_size = X_resampled.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(X_resampled)):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                       torch.tensor(y_train, dtype=torch.long))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters['batch_size'],
                                                   shuffle=True)

        model = ZDRN(input_size, hyperparameters['num_branches'], hyperparameters['num_layers'], num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'],
                               weight_decay=hyperparameters['weight_decay'])
        scheduler = OneCycleLR(optimizer, max_lr=hyperparameters['lr'],
                               total_steps=hyperparameters['total_iterations'])

        for epoch in range(hyperparameters['total_iterations']):
            train_loss = train_model(model, train_loader, criterion, optimizer, scheduler, device,
                                     hyperparameters['l1_lambda'])
            print(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')


if __name__ == "__main__":
    main()

