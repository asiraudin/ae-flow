# Projet AE-Flow

Article : https://openreview.net/forum?id=9OmCr1q54Z

## Datasets
- Chest X-Ray (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download)
- ...

## Utilisation
Les scripts de train et test sont dans le dossier src
- python3 train <submodel-name> <epochs> <model-path> <dataset-path>
    - <--submodel-name> "fast-flow" ou "res-net", default "fast-flow"
    - <--epochs> int, default 100
    - <--dataset-path> filepath to the data, default workdir/data/chest_xray.pch
    - <--model-path> filepath to save the trained model (saved after each epoch), default workdir/trained-model.pch

exemple : 
- python train.py
- python train.py --submodel_name res-net --epochs 20 --model_path ~/model-res-net.pch --dataset_path ~/data/chest_xray

- python3 test <submodel-name> <model-path> <dataset-path>
    - <--submodel-name> "fast-flow" ou "res-net", default "fast-flow" (has to correspond to the trained model)
    - <--dataset-path> filepath to the data, default workdir/data/chest_xray.pch
    - <--model-path> filepath to save the trained model (saved after each epoch), default workdir/trained-model.pch
exemple : 
- python test.py
- python test.py --submodel_name res-net --model_path ~/model-res-net.pch --dataset_path ~/data/chest_xray

## Données

La classe AEFlow dataset prend en charge le prétraitement des données
- les datasets doivent être placés dans le folder "data" avec la structure suivante
```
data .
    ├── mydataset                   # e.g chest_xray
        ├── raw                     # Données brutes
            ├── test                # Données de test
                ├── ANOMALY              # Images anormales
                └── NORMAL               # Images normales
            └── train               # Données de train
                └── NORMAL              # Images normales
    └── datasets.py
```
- Lorsque la classe AEFlow est instanciée, elle vérifie si les fichiers prétraités existe. Si non, elle réalise
le prétraitement, puis place les fichiers dans un dossier processed. Il y a 5 fichiers prétraités : 3 tenseurs
contenant les images de train et test (normales + anormales) + 2 tenseurs de labels pour les images de test.
```
data
    ├── mydataset                   # e.g chest_xray
        ├── raw                     # Données brutes
        └── processed               # Données prétraitées
            ├── test_ANOMALY.pt
            ├── test_ANOMALY_labels.pt
            ├── test_NORMAL.pt
            ├── test_NORMAL_labels.pt
            └── train_NORMAL.pt
    └── datasets.py
```

- Le stockage des données prétraitées permet de stocker les images de manière compact (256 x 256) + d'éviter d'avoir
à process le dataset à chaque fois (certains sont volumineux)


