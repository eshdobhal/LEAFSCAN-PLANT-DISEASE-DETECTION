| Test set | Images | Accuracy | Macro recall | Macro precision | Macro F1 | >=80% confident | Accuracy when confident |
|---|---|---|---|---|---|---|---|
| test_lab | 10937 | 99.41% | 99.25% | 99.32% | 99.28% | 98% | 99.87% |
| test_lab_tta | 10937 | 99.42% | 99.25% | 99.35% | 99.29% | 98% | 99.88% |
| test_field | 225 | 70.67% | 70.38% | 72.15% | 70.01% | 65% | 83.56% |
| test_field_tta | 225 | 70.67% | 70.10% | 71.77% | 69.83% | 61% | 87.59% |

Field recall per class (PlantDoc test, TTA):

| Class | Images | Recall |
|---|---|---|
| Tomato___Bacterial_spot | 8 | 25% |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 3 | 33% |
| Potato___Early_blight | 7 | 43% |
| Apple___Apple_scab | 10 | 50% |
| Cherry___healthy | 10 | 50% |
| Tomato___Leaf_Mold | 6 | 50% |
| Tomato___Tomato_mosaic_virus | 10 | 50% |
| Tomato___healthy | 8 | 50% |
| Tomato___Early_blight | 9 | 56% |
| Corn___Northern_Leaf_Blight | 10 | 60% |
| Tomato___Septoria_leaf_spot | 10 | 60% |
| Pepper,_bell___Bacterial_spot | 9 | 67% |
| Blueberry___healthy | 10 | 70% |
| Pepper,_bell___healthy | 8 | 75% |
| Apple___healthy | 9 | 78% |
| Apple___Cedar_apple_rust | 10 | 80% |
| Corn___Common_rust | 10 | 80% |
| Potato___Late_blight | 5 | 80% |
| Tomato___Late_blight | 10 | 80% |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5 | 80% |
| Soybean___healthy | 8 | 88% |
| Peach___healthy | 9 | 89% |
| Grape___Black_rot | 8 | 100% |
| Grape___healthy | 12 | 100% |
| Raspberry___healthy | 7 | 100% |
| Squash___Powdery_mildew | 6 | 100% |
| Strawberry___healthy | 8 | 100% |

Lowest lab recall: Corn___Cercospora_leaf_spot Gray_leaf_spot 88.8%, Tomato___Spider_mites Two-spotted_spider_mite 97.2%, Potato___Late_blight 97.6%, Tomato___Early_blight 97.7%, Apple___Apple_scab 98.4%
