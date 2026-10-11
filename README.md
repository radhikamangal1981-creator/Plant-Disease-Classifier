# 🌿 Plant Disease Classifier

Classifies apple, potato, and tomato leaf images into 5 classes using
transfer learning (MobileNetV2) in TensorFlow, deployed with Streamlit.

**Live demo:** [your-app-link](https://plant-disease-classifier-1.streamlit.app/)]

## Classes
Apple scab, Apple healthy, Potato early blight, Potato healthy, Tomato mosaic virus
(subset of the PlantVillage dataset)

## Models compared
| Model | Val Accuracy | Macro F1 | Val Loss |
|---|---|---|---|
| Custom CNN | 95.39% | 0.943 | 0.172 |
| MobileNetV2 (frozen) | 93.29% | 0.912 | 0.192 |
| MobileNetV2 ([aug + weights / aug + weights ]) | 99.21% | 0.987 | 0.021 |

All models use the same 80/20 train/validation split (seed 123).
The first frozen MobileNetV2 did not use the correct input scaling; the improved
model fixes this and adds augmentation and class weights, so the gain cannot be
attributed to any single change.

## Limitations
- Only 5 classes; the model has no "unknown" option and will give a label for any image.
- PlantVillage has clean backgrounds and near-duplicate photos, so validation
  accuracy is likely higher than real-world accuracy.
- Results come from one validation split and one training run per model, with no separate test set.
- Performance on real-world photos was not evaluated.

## Repo structure
- `app.py`: Streamlit app
- `notebooks/`: training and comparison notebooks
- `results/`: CSV results for each model

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
