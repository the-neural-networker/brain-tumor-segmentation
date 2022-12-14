import os 
import sys
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser

import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from src.losses import BCEDiceLoss

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)
    
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)

class UNet3d(pl.LightningModule):
    def __init__(self, in_channels=4, n_classes=3, n_channels=24, learning_rate=5e-4):
        super(UNet3d, self).__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

        self.criterion = BCEDiceLoss()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask

    def training_step(self, batch, batch_idx):
        X, y = batch['image'], batch['mask']
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        dice = self.dice_coef_metric(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_dice', dice, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        X, y = batch['image'], batch['mask']
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        dice = self.dice_coef_metric(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dice', dice, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 

    def test_step(self, batch, batch_idx) :
        X, y = batch['image'], batch['mask']
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        dice = self.dice_coef_metric(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_dice', dice, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        } 

    def dice_coef_metric(self, probabilities: torch.Tensor,
                        truth: torch.Tensor,
                        treshold: float = 0.5,
                        eps: float = 1e-9):
        """
        Calculate Dice score for data batch.
        Params:
            probobilities: model outputs after activation function.
            truth: truth values.
            threshold: threshold for probabilities.
            eps: additive to refine the estimate.
            Returns: dice score aka f1.
        """
        scores = []
        num = probabilities.shape[0]
        predictions = (probabilities >= treshold).float()
        assert(predictions.shape == truth.shape)
        for i in range(num):
            prediction = predictions[i]
            truth_ = truth[i]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores.append(1.0)
            else:
                scores.append(((intersection + eps) / union).item())
        return np.mean(scores)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_channels', type=int, default=4)
        parser.add_argument('--n_classes', type=int, default=3)
        parser.add_argument('--n_channels', type=int, default=24)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

if __name__ == '__main__':
    model = UNet3d()
    print(model)
    X = torch.randn((1, 4, 128, 128, 128))
    y_hat = model(X)
    print('output shape: ', y_hat.shape)