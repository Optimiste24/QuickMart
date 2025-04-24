import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import holidays

# 1. Chargement des données
def load_data():
    """Charge tous les fichiers CSV nécessaires"""
    df_sales = pd.read_csv('data/ventes.csv', parse_dates=['Date'])
    df_weather = pd.read_csv('data/meteo_locale.csv', parse_dates=['Date'])
    df_products = pd.read_csv('data/produits.csv')
    df_staff = pd.read_csv('data/plannings_equipes.csv', parse_dates=['Date'])
    
    return {
        'sales': df_sales,
        'weather': df_weather,
        'products': df_products,
        'staff': df_staff
    }

# 2. Nettoyage des données
def clean_data(df_sales, df_weather):
    """Nettoie et fusionne les données"""
    # Gestion des outliers (seuil à 99.5% pour conserver plus de données)
    q_high = df_sales['CA'].quantile(0.995)
    df_sales = df_sales[(df_sales['CA'] > 0) & (df_sales['CA'] < q_high)]
    
    # Détection des jours fermés (ventes = 0)
    daily_sales = df_sales.groupby('Date')['CA'].sum()
    closed_days = daily_sales[daily_sales == 0].index
    df_sales = df_sales[~df_sales['Date'].isin(closed_days)]
    
    # Fusion avec la météo
    df_merged = pd.merge(
        df_sales.groupby(['Date', 'Magasin'])['CA'].sum().reset_index(),
        df_weather,
        left_on=['Date', 'Magasin'],
        right_on=['Date', 'Ville'],
        how='left'
    )
    
    # Ajout des jours fériés
    fr_holidays = holidays.France(years=[2024])
    df_merged['is_holiday'] = df_merged['Date'].apply(lambda x: x in fr_holidays)
    
    return df_merged

# 3. Feature Engineering
def create_features(df):
    """Crée des variables supplémentaires"""
    # Variables temporelles
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Lag features (valeurs décalées)
    df['CA_lag7'] = df.groupby('Magasin')['CA'].shift(7)
    
    # Rolling means
    df['CA_rolling7'] = df.groupby('Magasin')['CA'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    
    return df.dropna()

# 4. Préparation pour Prophet
def prepare_for_prophet(df, magasin_id='Magasin_1'):
    """Prépare les données pour un magasin spécifique"""
    df_mag = df[df['Magasin'] == magasin_id].copy()
    
    prophet_df = df_mag[['Date', 'CA', 'Température']].rename(columns={
        'Date': 'ds',
        'CA': 'y',
        'Température': 'temperature'
    })
    
    # Ajout des variables supplémentaires
    prophet_df['is_weekend'] = df_mag['is_weekend']
    prophet_df['is_holiday'] = df_mag['is_holiday']
    
    return prophet_df

# 5. Entraînement du modèle
def train_prophet_model(df):
    """Entraîne un modèle Prophet avec regresseurs"""
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative'
    )
    
    # Ajout des regresseurs externes
    model.add_regressor('temperature')
    model.add_regressor('is_weekend')
    model.add_regressor('is_holiday')
    
    # Jours fériés français
    model.add_country_holidays(country_name='FR')
    
    model.fit(df)
    return model

# 6. Validation croisée
def cross_validate(model, df):
    """Évalue la performance avec une validation temporelle"""
    from prophet.diagnostics import cross_validation
    from prophet.diagnostics import performance_metrics
    
    df_cv = cross_validation(
        model,
        initial='60 days',
        period='15 days',
        horizon='30 days'
    )
    
    df_p = performance_metrics(df_cv)
    mape = df_p['mape'].mean()
    print(f"✅ MAPE moyen sur la validation croisée: {mape:.1%}")
    return mape < 0.15  # Seuil de 15%

# Pipeline complet
if __name__ == "__main__":
    print("🔍 Chargement des données...")
    data = load_data()
    
    print("🧹 Nettoyage et fusion des données...")
    df_clean = clean_data(data['sales'], data['weather'])
    
    print("⚙️ Création des features...")
    df_features = create_features(df_clean)
    
    print("📊 Préparation pour Prophet...")
    prophet_data = prepare_for_prophet(df_features)
    
    print("🤖 Entraînement du modèle...")
    model = train_prophet_model(prophet_data)
    
    print("🧪 Validation du modèle...")
    if cross_validate(model, prophet_data):
        print("💾 Sauvegarde du modèle...")
        from prophet.serialize import model_to_json
        with open('models/prophet_model.json', 'w') as f:
            f.write(model_to_json(model))
        print("✅ Prétraitement terminé et modèle sauvegardé!")
    else:
        print("❌ Le modèle ne respecte pas le seuil de performance")