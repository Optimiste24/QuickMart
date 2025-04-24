# QuickMart Sales Forecasting App

Ce script est une application web interactive développée avec **Streamlit** destinée à la prévision des ventes pour une chaîne de magasins fictive nommée "QuickMart". 

## Fonctionnalités principales

### 1. Chargement et validation des données
- Charge les données depuis 3 fichiers CSV :
  - `ventes_enhanced.csv` (données de ventes)
  - `meteo_locale.csv` (données météo)
  - `campagnes_marketing.csv` (promotions)
- Validation des colonnes nécessaires
- Filtrage sur la période janvier-mars 2024
- Gestion robuste des erreurs

### 2. Préparation des données
- Agrégation des ventes journalières
- Fusion avec les données météo
- Imputation des valeurs manquantes
- Ajout de variables temporelles :
  - Jours de semaine/weekend
  - Jours fériés (via librairie `holidays`)
  - Périodes promotionnelles

### 3. Modélisation avec Prophet
- Modèle de séries temporelles avec :
  - Saisonnalité hebdomadaire
  - Régresseurs externes (température, promotions, etc.)
  - Paramètres ajustables
- Prévision sur 30 jours
- Validation croisée (MAPE, RMSE)

### 4. Interface Streamlit
**Barre latérale** : Configuration du modèle

**Onglets** :
- Dashboard : KPI et visualisations
- Analyse : Prévisions et composantes du modèle
- Export : Téléchargement des résultats (CSV, Excel, JSON)
