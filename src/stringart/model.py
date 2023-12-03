import lightning as L
import numpy as np
import torch

LEARNING_RATE=1e-3


class ClippedLinearModel(L.LightningModule):
    
    def __init__(self, in_features: int):
        self.linear = torch.nn.Linear(in_features, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer