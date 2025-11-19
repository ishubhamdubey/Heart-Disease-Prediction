# Heart Attack Risk Prediction from Retinal Images

An end-to-end (prototype) machine learning pipeline to estimate heart attack risk using retinal (fundus) images. The current version supports:

## What’s Inside
- Modular data loader (`src/data/dataset.py`) with fallback to simulated tabular data if images are absent.
- CNN feature extractor & classifier (`src/models/cnn.py`).
- Traditional ML models (AdaBoost) and unsupervised clustering (KMeans) built on either raw tabular data or CNN-extracted features.
- Unified training orchestrator (`src/training/train.py`).
- CLI entry point (`main.py`).

## Folder Structure (after setup)
```
project/
	data/
		retina/
			healthy/*.jpg|png
			risk/*.jpg|png
	src/
		data/dataset.py
		models/cnn.py
		training/train.py
	main.py
```

## Getting Started
1. Create & activate a virtual environment (recommended).
2. Install dependencies:
	 `pip install -r requirements.txt`
3. Add retinal images into `data/retina/healthy/` and `data/retina/risk/` (binary labels: healthy=0, risk=1). If these folders are empty or missing, simulated data is used.
4. Run training:
	 `python main.py --data data/retina --epochs 8`

### Quick Demo (Manual Sample Images)
Obtain 1–2 publicly available retina images (normal + pathology) and place them manually:
```
data/retina/healthy/normal1.jpg
data/retina/risk/pathology1.jpg
```
Then run:
```
python main.py --data data/retina --epochs 2
python -m src.inference.predict data/retina/healthy/normal1.jpg
```
Resulting probabilities are NOT medically meaningful with such a tiny set—use a proper labeled dataset.

## Inference (Single Image)
After training, run:
`python -m src.inference.predict path/to/retina_image.jpg --artifacts artifacts`
Returns JSON-like dict with CNN + AdaBoost probabilities.

Artifacts saved to `artifacts/`:
- `cnn_model.h5`
- `ada_model.pkl`
- `kmeans.pkl`

## CLI Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `data` | Root folder containing `healthy/` and `risk/` subfolders |
| `--epochs` | 5 | CNN training epochs |

## Pipeline Overview
1. Load images (or simulate data if none found).
2. Train CNN (if image data) and extract penultimate-layer features.
3. Train AdaBoost classifier on features.
4. Run KMeans clustering for exploratory grouping.
5. Print classification reports and cluster distribution.

## Extending the Project
Planned / suggested improvements:
- Add data augmentation & preprocessing (contrast enhancement, vessel segmentation).
- Integrate transfer learning (e.g., EfficientNet / ResNet) for better baseline performance.
- Add evaluation metrics beyond accuracy (AUC, F1, sensitivity/specificity).
- Provide model persistence (save `model.h5`, `ada.pkl`).
- Add inference script for single image prediction.
- Implement explainability (Grad-CAM) to highlight retinal regions influencing predictions.

## Disclaimer
This is a prototype. Medical usage requires regulatory approval, high-quality curated datasets, and rigorous validation.

## License
MIT (add LICENSE file if distributing).

## Acknowledgements
Inspired by research linking retinal microvascular features and cardiovascular risk.
