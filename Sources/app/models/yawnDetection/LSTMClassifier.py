import torch
import torch.nn as nn
import torchmetrics
import numpy as np
import pytorch_lightning as pl

class LSTMClassifier(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''
    def __init__(self, 
                 n_features, 
                 hidden_size, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 dropout, 
                 learning_rate,
                 criterion):
        super(LSTMClassifier, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        if criterion == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Unknown loss criterion")
        
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        

        self.model = nn.Sequential(
            nn.Linear(hidden_size, 30),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(30, 2),
            # nn.Softmax(dim=1),
        )

        self.train_metrics = nn.ModuleDict({
            'train_acc': torchmetrics.classification.MulticlassAccuracy(2),
            'train_f1': torchmetrics.classification.F1Score(task='multiclass', num_classes=2, average='macro'),
            'train_P': torchmetrics.classification.Precision(task='multiclass', num_classes=2, average='macro'),
            'train_R': torchmetrics.classification.Recall(task='multiclass', num_classes=2, average='macro'),
        })

        self.val_metrics = nn.ModuleDict({
            'val_acc': torchmetrics.classification.MulticlassAccuracy(2),
            'val_f1': torchmetrics.classification.F1Score(task='multiclass', num_classes=2, average='macro'),
            'val_P': torchmetrics.classification.Precision(task='multiclass', num_classes=2, average='macro'),
            'val_R': torchmetrics.classification.Recall(task='multiclass', num_classes=2, average='macro'),
        })

        self.test_metrics = nn.ModuleDict({
            'test_acc': torchmetrics.classification.MulticlassAccuracy(2),
            'test_f1': torchmetrics.classification.F1Score(task='multiclass', num_classes=2, average='macro'),
            'test_P': torchmetrics.classification.Precision(task='multiclass', num_classes=2, average='macro'),
            'test_R': torchmetrics.classification.Recall(task='multiclass', num_classes=2, average='macro'),
        })

        self.save_hyperparameters()
        
    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.model(lstm_out[:,-1])
        return y_pred
    
    def predict(self, x):
        with torch.no_grad():
            y_hat = self(x)
            preds = nn.functional.softmax(y_hat, dim=1).detach().cpu().numpy()
        
        return preds
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = nn.functional.softmax(y_hat, dim=1)
        
        for k, metric in self.train_metrics.items():
            metric(preds, y)
            self.log(k, metric, on_step=True, on_epoch=True)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = nn.functional.softmax(y_hat, dim=1)
        
        for k, metric in self.val_metrics.items():
            metric(preds, y)
            self.log(k, metric, on_step=True, on_epoch=True)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.criterion(y_hat, y)
    #     preds = nn.functional.softmax(y_hat, dim=1)
        
    #     for k, metric in self.test_metrics.items():
    #         metric(preds, y)
    #         self.log(k, metric, on_step=True, on_epoch=True)
        
    #     self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
    #     return loss
