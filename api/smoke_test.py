import pickle
import unittest
from pathlib import Path

import numpy as np
import torch

from main import MODEL_PATH, SCALER_PATH, FishMortalityLSTM, predict_readings


class FiveMinuteModelSmokeTest(unittest.TestCase):
    def test_checkpoint_scaler_and_window_inference(self):
        self.assertTrue(MODEL_PATH.is_file())
        self.assertTrue(SCALER_PATH.is_file())

        with SCALER_PATH.open("rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        model = FishMortalityLSTM()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()

        reading = np.array([[7.0, 27.0, 7.5, 15.0, 0.1, 300.0]], dtype=np.float32)
        scaled = scaler.transform(reading)
        sequence = torch.tensor(np.repeat(scaled, 144, axis=0), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(sequence)

        self.assertEqual(tuple(output.shape), (1, 3))
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()