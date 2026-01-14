#  Projet Big Data : NYC Taxi Prediction Pipeline

**Étudiant :** Ben Amor Oussama
**Cours :** Big Data 

## Présentation
Ce projet implémente un pipeline Big Data complet pour analyser et prédire le prix des courses de taxi à New York.

Il valide les compétences suivantes :
Nettoyage et structuration des données.
Entraînement d'un modèle de Régression Linéaire.
Déploiement du modèle pour des prédictions en temps réel.

## Architecture
* **Langage :** Python (PySpark)
* **Stockage :** Parquet (Optimisé)
* **Input Streaming :** Surveillance de dossier (Hot Folder)

## Comment lancer le projet :

### . Analyse et Nettoyage :
```bash
python src/1_analyse_sql.py

### . Entraînement de l'IA :
python src/2_entrainement_ml.py

### . Lancement du Streaming :
python src/3_streaming_reel.py