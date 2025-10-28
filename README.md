# Advanced Audio Classifier  
**Author:** itznan  

An advanced machine learning pipeline for classifying audio samples using feature extraction (MFCCs, spectral features, chroma, tempo, etc.) and ensemble learning (Random Forest, Gradient Boosting, SVM).  
This project includes both a **training pipeline** and a **user-friendly prediction interface**.

---

## Project Structure
```

.
├── train.py           # Full training and evaluation pipeline
├── use.py             # Easy-to-use prediction interface
├── Data/              # Directory containing audio files organized in subfolders by class
├── voice_classifier.pkl  # Saved trained model (generated after training)
└── requirements.txt   # Python dependencies

````

---

## Features
Automatic feature extraction from audio files  
Data augmentation (time stretch, pitch shift, noise)  
Caching system for faster feature reuse  
Model optimization with GridSearchCV  
Ensemble model combining Random Forest, Gradient Boosting, and SVM  
Detailed logs, metrics, and confusion matrix output  
User-friendly CLI for predictions and batch processing  

---

## Requirements
Install all dependencies:
```bash
pip install -r requirements.txt
````

### Dependencies include:

* numpy
* librosa
* scikit-learn
* joblib
* PyYAML
* tqdm
* soundfile

---

## Dataset Structure

Your audio data should be placed in the `Data/` directory, organized as follows:

```
Data/
├── class_1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── class_2/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```

Each subdirectory name represents a class label.

---

## Training the Model

To train the model:

```bash
python train.py --data-dir Data
```

### Optional Arguments:

| Argument       | Description              |
| -------------- | ------------------------ |
| `--config`     | Path to YAML config file |
| `--model-path` | Custom model save path   |
| `--limit`      | Limit samples per class  |
| `--augment`    | Enable data augmentation |
| `--debug`      | Enable debug logging     |
| `--no-cache`   | Disable feature caching  |

Example:

```bash
python train.py --data-dir Data --augment --limit 100
```

After training, a model file `voice_classifier.pkl` will be saved automatically.

---

## Using the Model

You can classify audio using the **use.py** script.

### Predict a Single Audio File

```bash
python use.py path/to/audio.wav
```

### Predict Multiple Files

```bash
python use.py audio1.wav audio2.wav audio3.wav
```

### Predict All Files in a Directory

```bash
python use.py --directory ./test_audio/
```

### Recursive Prediction and JSON Output

```bash
python use.py --directory ./test_audio/ --recursive --output results.json
```

### Show Model Information

```bash
python use.py --model voice_classifier.pkl --info
```

---

## 🧾 Example Output

```
Predicted Class: female_voice
Confidence: 92.47%
[█████████████████████████░░░░░░░░░░░░]
```

---

## Model Details

* **Feature extraction:** MFCC, spectral centroid, bandwidth, rolloff, chroma, ZCR, energy, tempo
* **Models used:**

  * Random Forest (optimized via GridSearch)
  * Gradient Boosting
  * Support Vector Machine (for smaller datasets)
* **Ensemble:** Weighted soft voting classifier

---

## Logs and Outputs

During training, detailed logs are saved as:

```
training_YYYYMMDD_HHMMSS.log
```

This includes:

* Training/validation/test accuracy
* Best hyperparameters
* Confusion matrix
* Top features by importance

---

## Saving and Loading

The model and preprocessing scaler are stored in:

```
voice_classifier.pkl
```

This file contains:

* Trained ensemble model
* Individual models (RF, GB, SVM)
* Feature scaler
* Class mappings
* Training metadata

---

## Author

**itznan**

> Built for experimentation, learning, and advanced audio classification research.

---

## License

This project is open for educational and research purposes.
Feel free to modify and extend it — just credit **itznan** when you do.

---
