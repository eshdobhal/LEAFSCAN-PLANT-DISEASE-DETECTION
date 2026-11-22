| Test set | Images | Accuracy | Macro recall | Macro precision | Macro F1 | >=80% confident | Accuracy when confident |
|---|---|---|---|---|---|---|---|
| test_lab | 10937 | 96.13% | 95.74% | 95.13% | 95.16% | 82% | 99.32% |
| test_lab_tta | 10937 | 96.48% | 96.15% | 95.59% | 95.64% | 82% | 99.55% |
| test_field | 225 | 69.78% | 68.57% | 71.57% | 68.55% | 40% | 87.78% |
| test_field_tta | 225 | 72.00% | 70.98% | 74.19% | 70.83% | 40% | 87.64% |

Field recall per class (PlantDoc test, TTA):

| Class | Images | Recall |
|---|---|---|
| Potato___Early_blight | 7 | 29% |
| Tomato___Tomato_mosaic_virus | 10 | 30% |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 3 | 33% |
| Tomato___Bacterial_spot | 8 | 38% |
| Pepper,_bell___Bacterial_spot | 9 | 44% |
| Cherry___healthy | 10 | 50% |
| Tomato___healthy | 8 | 50% |
| Corn___Northern_Leaf_Blight | 10 | 60% |
| Potato___Late_blight | 5 | 60% |
| Tomato___Leaf_Mold | 6 | 67% |
| Apple___Apple_scab | 10 | 70% |
| Tomato___Septoria_leaf_spot | 10 | 70% |
| Soybean___healthy | 8 | 75% |
| Apple___healthy | 9 | 78% |
| Peach___healthy | 9 | 78% |
| Tomato___Early_blight | 9 | 78% |
| Apple___Cedar_apple_rust | 10 | 80% |
| Corn___Common_rust | 10 | 80% |
| Tomato___Late_blight | 10 | 80% |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5 | 80% |
| Pepper,_bell___healthy | 8 | 88% |
| Blueberry___healthy | 10 | 100% |
| Grape___Black_rot | 8 | 100% |
| Grape___healthy | 12 | 100% |
| Raspberry___healthy | 7 | 100% |
| Squash___Powdery_mildew | 6 | 100% |
| Strawberry___healthy | 8 | 100% |

Lowest lab recall: Tomato___Early_blight 75.8%, Tomato___Spider_mites Two-spotted_spider_mite 77.2%, Potato___Late_blight 83.5%, Tomato___Target_Spot 86.5%, Corn___Northern_Leaf_Blight 90.5%
