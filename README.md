# 🌿 Plant Disease Classifier

Classifies apple, potato, and tomato leaf images into 5 classes using transfer learning (MobileNetV2) in TensorFlow, and compares it with a custom CNN baseline. Deployed as a Streamlit web app.

**Live demo:** YOUR-APP-[LINK](https://plant-disease-classifier-1.streamlit.app/)

![App screenshot](<img width="772" height="821" alt="image" src="https://github.com/user-attachments/assets/ba41881e-db09-4958-9d33-133a3b170398" />
)

## Dataset

A 5-class subset of the [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) dataset (3,800 images, resized to 180x180):

- Apple: Apple scab
- Apple: Healthy
- Potato: Early blight
- Potato: Healthy
- Tomato: Tomato mosaic virus

The data was split 80/20 into training (3,040 images) and validation (760 images) sets, with the same seed (123) for every model.

## Models compared

| Model | Val Accuracy | Macro F1 | Val Loss |
|---|---|---|---|
| Custom CNN | 95.39% | 0.943 | 0.172 |
| MobileNetV2 (frozen) | 93.29% | 0.912 | 0.192 |
| MobileNetV2 (aug + weights) | 99.21% | 0.987 | 0.021 |

- **Custom CNN:** a small convolutional network (three Conv2D blocks) trained from scratch.
- **MobileNetV2 (frozen):** pretrained ImageNet base kept frozen, with a new classification head trained on top.
- **MobileNetV2 (aug + weights):** the same frozen base, with the correct MobileNetV2 input scaling, data augmentation (flip, rotation, zoom), and class weights for the imbalanced classes. The base model was **not** fine-tuned.

The first frozen MobileNetV2 did not use the correct input scaling, while the improved model does and also adds augmentation and class weights. These changes were made together, so the gain cannot be attributed to any single one.

## Limitations

- Only 5 classes. The model has no "unknown" option, so it returns one of the 5 labels for any image, even a non-leaf.
- PlantVillage images have clean backgrounds, and the dataset contains near-duplicate photos of the same leaf. Validation accuracy is therefore likely higher than real-world accuracy.
- Results come from a single validation split and one training run per model. There is no separate test set.
- Performance on real-world photos (different backgrounds and lighting) was not evaluated.
- The app is a demo, not a substitute for expert diagnosis.

## Repository structure

```
app.py                       Streamlit app
plant_disease_model.keras    trained model (MobileNetV2, aug + weights)
requirements.txt             dependencies
notebooks/                   training and comparison notebooks
results/                     CSV results for each model
```

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Possible next steps

- Evaluate on real-world photos (for example PlantDoc) and report the drop in accuracy
- Add Grad-CAM to check whether the model focuses on the lesion or the background
- Add a fine-tuned MobileNetV2 variant
- Add more classes and an "unknown" option

## Tech

Python, TensorFlow/Keras, scikit-learn, Streamlit
