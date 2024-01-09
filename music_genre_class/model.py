from typing import Any

import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class CNNModel(pl.LightningModule):
    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.example_input_array = torch.rand((4, 1, 100, 100))
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            num_classes=10, task="multiclass"
        )
        self.f1_score = torchmetrics.classification.F1Score(
            num_classes=10, task="multiclass", average="macro"
        )
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 23 * 23, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.LeakyReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        spec, labels = batch
        y_preds = self(spec)
        loss = self.loss_fn(y_preds, labels)
        acc = self.accuracy(y_preds, labels)
        f1 = self.f1_score(y_preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_f1", f1, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        spec, labels = batch
        y_preds = self(spec)
        loss = self.loss_fn(y_preds, labels)
        acc = self.accuracy(y_preds, labels)
        f1 = self.f1_score(y_preds, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_f1", f1, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss}

    def configure_optimizers(self) -> Any:
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config["train"]["weight_decay"],
            }
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config["train"]["learning_rate"],
        )

        return [optimizer]
