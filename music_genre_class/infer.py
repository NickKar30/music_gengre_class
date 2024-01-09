import hydra
import numpy as np
import onnxruntime
import pandas as pd
from omegaconf import DictConfig

from dataset import AudioDataModule


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Load the saved onnx model
    onnx_file_path = "data/onnx/model.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_file_path)

    # Prepare the test data
    dm = AudioDataModule(
        dataset_path=cfg.data.dataset_path,
        val_size=cfg.data.val_size,
        batch_size=cfg.data.batch_size,
        genres=cfg.data.genres,
        num_waves=cfg.data.num_waves,
    )
    dm.setup()
    test_data = dm.val_dataloader()

    for batch in test_data:
        spectrogram, labels = batch
        input_name = ort_session.get_inputs()[0].name
        logits = ort_session.run(None, {input_name: spectrogram.numpy()})
        predictions = np.argmax(logits[0], axis=1)
    genres = cfg.data.genres
    genre_dict = {i: genre for i, genre in enumerate(genres)}

    df = pd.DataFrame({"predictions": predictions, "label": labels})
    df["predictions"] = df["predictions"].map(genre_dict)
    df["label"] = df["label"].map(genre_dict)
    df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
