# music_gengre_class

## Task
The goal of the project is to classify musical genres into 10 categories, based on the analysis of audio files.

## Data
The data for the project are taken from [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification):

 The dataset consists of 1000 audio files of different lengths, 100 of 10 music genre:

 - blues

 - classical

 - country

 - disco

 - hiphop

 - jazz

 - metal

 - pop

 - reggae

 - rock

By default, the value of downloading 5 audio files from each genre is set to reduce the data loading time. If necessary, this number can be changed in the ``config``.

## Solution
As a solution to the problem, a convolutional neural network was used, which was fed with spectrograms - visual representations of audio signals, which show the distribution of frequency components of the signal over time.

The obtained spectrograms were compressed to the size of 100*100 pixels, to speed up the calculations and reduce the memory size. This function can be disabled if higher image quality is required.

The compressed spectrograms were fed to the input of the convolutional neural network, which performed the classification of audio files by genres.

## Dev tools

- ``poetry`` for creating and installing Python packages
  
- ``pre-commit`` for checking and formatting code before committing
  
- ``DVC`` for versioning data and models and reproducing experiments
  
- ``Hydra`` for managing project configuration and combining config files
  
- ``MLflow`` for logging hyperparameters and metrics 
  
- ``ONNX`` for storing and distributing models between different frameworks

## Project structure

- ``data/``: contains .dvc files for train and test sets;
- ``.dvc/``: service files and folders for DVC
- ``conf/``: contains yaml file config with project settings and training hyperparameters models
- ``models/``: содержит .dvc-файл предобученной модели; в нее же сохраняется обученная модель в случае запуска скрипта train.py
- ``music_genre_class``: python package that contains:
  - ``dataset.py``: creation and preprocessing of downloaded audio files
  - ``model.py``: model architecture
