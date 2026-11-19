| Test set | Images | Accuracy | Macro recall | Macro precision | Macro F1 | >=80% confident | Accuracy when confident |
|---|---|---|---|---|---|---|---|
| test_lab | 10937 | 99.63% | 99.53% | 99.44% | 99.49% | 99% | 99.86% |
| test_lab_tta | 10937 | 99.63% | 99.53% | 99.45% | 99.49% | 99% | 99.85% |
| test_field | 225 | 70.22% | 69.46% | 72.16% | 68.48% | 68% | 81.05% |
| test_field_tta | 225 | 71.11% | 70.36% | 72.86% | 69.27% | 63% | 83.69% |

Field recall per class (PlantDoc test, TTA):

| Class | Images | Recall |
|---|---|---|
| Tomato___Tomato_mosaic_virus | 10 | 20% |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 3 | 33% |
| Tomato___Bacterial_spot | 8 | 38% |
| Potato___Early_blight | 7 | 43% |
| Cherry___healthy | 10 | 50% |
| Tomato___Leaf_Mold | 6 | 50% |
| Tomato___healthy | 8 | 50% |
| Tomato___Early_blight | 9 | 56% |
| Blueberry___healthy | 10 | 60% |
| Corn___Northern_Leaf_Blight | 10 | 60% |
| Soybean___healthy | 8 | 62% |
| Apple___Cedar_apple_rust | 10 | 70% |
| Tomato___Late_blight | 10 | 70% |
| Pepper,_bell___healthy | 8 | 75% |
| Apple___healthy | 9 | 78% |
| Potato___Late_blight | 5 | 80% |
| Tomato___Septoria_leaf_spot | 10 | 80% |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5 | 80% |
| Grape___Black_rot | 8 | 88% |
| Peach___healthy | 9 | 89% |
| Pepper,_bell___Bacterial_spot | 9 | 89% |
| Apple___Apple_scab | 10 | 90% |
| Corn___Common_rust | 10 | 90% |
| Grape___healthy | 12 | 100% |
| Raspberry___healthy | 7 | 100% |
| Squash___Powdery_mildew | 6 | 100% |
| Strawberry___healthy | 8 | 100% |

Lowest lab recall: Corn___Cercospora_leaf_spot Gray_leaf_spot 94.4%, Corn___Northern_Leaf_Blight 97.3%, Peach___healthy 98.6%, Tomato___Early_blight 98.6%, Tomato___Target_Spot 98.6%

[exited with code 0]
