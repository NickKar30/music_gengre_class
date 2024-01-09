import logging
import os
from typing import Optional, Tuple

import librosa
import pytorch_lightning as pl
import torch
import torchaudio
from dvc.api import DVCFileSystem
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


class AudioDataset(Dataset):
    def __init__(self, data: dict, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data["spec"])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        spec = self.data["spec"][idx]
        label = self.data["labels"][idx]
        if self.transform:
            spec = self.transform(spec)
        return spec, label


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        url: str,
        val_size: float,
        batch_size: int,
        num_waves: int,
        genres: list,
        sample_rate: int = 22050,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_path = dataset_path
        self.url = url
        self.val_size = val_size
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.genres = genres
        self.num_waves = num_waves
        self.train_dataset = []
        self.val_dataset_dataset = []

    def prepare_data(self):
        fs = DVCFileSystem(self.url, rev="main")
        os.makedirs("data/genres_original")
        for genre in self.genres:
            logging.info(f"Loading: {genre}")
            subfolder = os.path.join("data", "genres_original", genre)
            os.mkdir(subfolder)
            files = fs.glob(f"data/data/genres_original/{genre}/*.wav")
            for file in files[: self.num_waves]:
                fs.get(file, os.path.join(subfolder, file.split("/")[-1]))
        logging.info("The data loading is complete!")

    def setup(self, stage: Optional[str] = None):
        self.data = create_data_dict("data/genres_original", self.sample_rate)
        train_spec, val_spec, train_labels, val_labels = train_test_split(
            self.data["spec"], self.data["labels"], test_size=self.val_size
        )
        train_dataset = AudioDataset({"spec": train_spec, "labels": train_labels})
        val_dataset = AudioDataset({"spec": val_spec, "labels": val_labels})
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )


def create_data_dict(dataset_path: str, sample_rate: int = 22050) -> dict:
    data = {"mapping": [], "spec": [], "labels": []}
    featurizer = torchaudio.transforms.Spectrogram(
        n_fft=1024, win_length=1024, hop_length=512, center=True
    )
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_comp = dirpath.split("/")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)
            logging.info(f"Processing: {semantic_label}")
            for f in filenames:
                if f == "jazz.00054.wav":
                    # skip file more 1 Mb
                    continue
                else:
                    file_path = os.path.join(dirpath, f)
                    audio, sr = librosa.load(file_path, sr=sample_rate)
                    audio = torch.Tensor(audio[: sample_rate * 10])
                    spec = torch.log(featurizer(audio)).clamp(1e-6)
                    data["spec"].append(spec)
                    data["labels"].append(i - 1)
    transform = transforms.Compose([transforms.Resize((100, 100))])
    for i in range(len(data["spec"])):
        data["spec"][i] = transform(data["spec"][i].unsqueeze(0))

    return data
