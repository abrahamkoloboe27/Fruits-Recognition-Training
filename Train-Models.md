# Préparation des Données et Entraînement des Modèles**

---

### **Introduction**  
Ce projet vise à entraîner et comparer plusieurs modèles de deep learning pour classer les images issues du dataset [Fruits 360 (100x100)](https://github.com/fruits-360/fruits-360-100x100). L’objectif est de sélectionner le **modèle le plus performant** selon plusieurs métriques pour l’envoyer en production. Ce README documente la **préparation des données**, l’**entraînement des modèles** et la **méthodologie utilisée**.

---

### **Table des Matières**  
1. [Dataset et Préparation](#dataset-et-préparation)  
2. [Data Augmentation avec Albumentations](#data-augmentation-avec-albumentations)  
3. [Architecture des Modèles Entraînés](#architecture-des-modèles-entrainés)  
4. [Gestion des Callbacks et Enregistrement des Modèles](#gestion-des-callbacks-et-enregistrement-des-modèles)  
5. [Suivi des Performances et Calcul des Métriques](#suivi-des-performances-et-calcul-des-métriques)  
6. [Références et Ressources Utiles](#références-et-ressources-utiles)

---

### **1. Dataset et Préparation**  
Nous utilisons le **dossier `Training`** du dataset Fruits 360, contenant des images de différentes classes de fruits.  
- **Split des données :** 75 % pour l’entraînement et 25 % pour la validation.  
- **Pourquoi ce split ?** Il permet de s’assurer que le modèle peut bien généraliser sur des données non vues.

#### **Commandes pour Préparer le Dataset**  
```bash
# Clone du repo et déplacement dans le répertoire
git clone https://github.com/fruits-360/fruits-360-100x100.git
cd fruits-360-100x100

# Split du dataset
python scripts/split_data.py --input_dir Training --split_ratio 0.25
```

---

### **2. Data Augmentation avec Albumentations**  
Pour **enrichir le dataset** et éviter l’overfitting, nous avons appliqué de la **data augmentation** avec [Albumentations](https://albumentations.ai/).  
**Transformations appliquées :**  
- **Rotation** aléatoire entre -15° et +15°  
- **Flip horizontal**  
- **Modification de la luminosité et du contraste**  

#### **Extrait de Code : Data Augmentation**
```python
import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.pytorch import ToTensorV2

# Définition de l'augmentation
transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.HueSaturationValue(),
    ], p=0.5),
    ToTensorV2(),
])
```

**Pourquoi utiliser Albumentations ?**  
- Optimisé pour des performances rapides.  
- Prend en charge **PyTorch et TensorFlow/Keras**.  

Pour en savoir plus : [Guide Albumentations](https://albumentations.ai/docs/).

---

### **3. Architecture des Modèles Entraînés**  
Nous avons testé 4 architectures :  
- **CNN Custom** (baseline simple)  
- **ResNet50**  
- **VGG16**  
- **EfficientNetB0**  

<div style="text-align: center;">
<table style="margin: auto;">
<tr>
<th>CNN Custom</th>
<th>EfficientNet</th>
<th>ResNet</th>
<th>VGG16</th>
</tr>
<tr>
<td>
<img src="images-models/svg/CNN.svg" alt="CNN Custom" width="100"/>
</td>
<td>
<img src="images-models/svg/EfficientNet-Base.svg" alt="EfficientNet" width="100"/>
</td>
<td>
<img src="images-models/svg/ResNet-Fine-Tuning.svg" alt="ResNet" width="100"/>
</td>
<td>
<img src="images-models/svg/VGG16-Fine-Tuning.svg" alt="VGG16" width="100"/>
</td>
</tr>
</table>
</div>


#### **Pourquoi plusieurs modèles ?**  
- **ResNet** et **EfficientNet** offrent des performances élevées sur des tâches de classification.  
- **VGG16** est plus simple mais toujours compétitif.  
- **CNN Custom** permet d’établir une **baseline** à partir de laquelle comparer les performances.

**Code : Initialisation des Modèles Pré-entrainés**  
```python
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0

model = ResNet50(weights='imagenet', input_shape=(100, 100, 3), include_top=False)
model.trainable = True  # Fine-tuning
```

---

### **4. Gestion des Callbacks et Enregistrement des Modèles**  
Nous avons utilisé un **callback Keras** pour enregistrer uniquement le **meilleur modèle** sur la validation.

#### **Extrait de Code : Callback**
```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.keras', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max'
)
```

**Pourquoi utiliser un callback ?**  
- Automatisation de l’enregistrement du **meilleur modèle** pour éviter les surentraînements.  
- Permet de **recharger facilement** le modèle pour une utilisation future.  

---

### **5. Suivi des Performances et Calcul des Métriques**  
Après l’entraînement, nous avons calculé plusieurs métriques pour évaluer les performances sur la validation :  
- **F1-Score** : Équilibre entre précision et rappel.  
- **AUC (Area Under Curve)** : Mesure de la capacité du modèle à distinguer les classes.  
- **Précision et Rappel** : Pour évaluer la qualité des prédictions.

#### **Extrait de Code : Calcul des Métriques**
```python
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred.argmax(axis=1), average='weighted')
auc = roc_auc_score(y_val, y_pred, multi_class='ovo')
precision = precision_score(y_val, y_pred.argmax(axis=1), average='weighted')
recall = recall_score(y_val, y_pred.argmax(axis=1), average='weighted')

print(f"F1: {f1}, AUC: {auc}, Precision: {precision}, Recall: {recall}")
```

---

### **6. Références et Ressources Utiles**  
- **Albumentations :** [Documentation](https://albumentations.ai/docs/)  
- **Keras Callbacks :** [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/)  
- **ResNet et Fine-Tuning :** [Article de référence](https://arxiv.org/abs/1512.03385)  
- **EfficientNet :** [Article de recherche](https://arxiv.org/abs/1905.11946)  
- **Introduction aux métriques ML :** [Guide Sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

### **Comment Répliquer l’Entraînement ?**  
Pour reproduire l’entraînement :  
1. **Cloner le repo :**  
   ```bash
   git clone <url_du_repo>
   cd <nom_du_repo>
   ```
2. **Installer les dépendances :**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Lancer l’entraînement :**  
   ```bash
   python train_model.py
   ```

---

### **Conclusion**  
Ce README documente toutes les étapes de **préparation des données** et **entraînement des modèles**. Chaque décision technique a été justifiée pour **assurer la qualité du modèle final**. N'hésitez pas à explorer les **références fournies** pour approfondir votre compréhension des concepts utilisés.

