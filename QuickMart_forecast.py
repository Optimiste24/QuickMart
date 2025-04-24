import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import holidays
import warnings

# Configuration
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="QuickMart Forecast Pro",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre amélioré
st.title("🛒 QuickMart Forecast Pro")
st.markdown("""
**Outil prédictif des ventes** intégrant météo, promotions et analyse des performances magasins  
*Période : Janvier - Mars 2024*  
""")

# ---- 1. Chargement des données optimisé ---- 
@st.cache_data(ttl=3600, show_spinner="Chargement des données...")
def load_data():
    """Charge et valide les données avec gestion robuste des erreurs"""

    data_dir = Path('data')
    
    try:
        # Définition des colonnes requises et à parser comme dates
        required_cols = {
            'ventes_enhanced.csv': {
                'columns': ['Date', 'Magasin', 'CA'],
                'parse_dates': ['Date']
            },
            'meteo_locale.csv': {
                'columns': ['Date', 'Température'],
                'parse_dates': ['Date']
            },
            'campagnes_marketing.csv': {
                'columns': ['Date_début', 'Date_fin', 'Type'],
                'parse_dates': ['Date_début', 'Date_fin']
            }
        }

        dfs = {}
        for file, params in required_cols.items():
            file_path = data_dir / file
            df = pd.read_csv(
                file_path, 
                encoding='latin1', 
                parse_dates=params['parse_dates']
            )

            missing_cols = [col for col in params['columns'] if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes dans {file} : {missing_cols}")

            dfs[file.replace('.csv', '')] = df

        # Filtrage de la période sur les ventes
        start_date, end_date = pd.to_datetime('2024-01-01'), pd.to_datetime('2024-03-31')
        ventes = dfs['ventes_enhanced']
        ventes = ventes[(ventes['Date'] >= start_date) & (ventes['Date'] <= end_date)]
        dfs['ventes_enhanced'] = ventes

        return dfs

    except Exception as e:
        st.error(f"""**Erreur critique** : {str(e)}  
                Vérifiez que :  
                - Les fichiers sont dans le dossier `data`  
                - Les colonnes requises sont présentes""")
        st.stop()


# ---- 2. Préparation des données avancée ----
def enhance_data(df_sales, df_weather, df_marketing):
    """Feature engineering complet"""
    # Agrégation multi-niveaux
    df_daily = df_sales.groupby('Date').agg({
        'CA': ['sum', 'mean', 'count'],
        'Magasin': pd.Series.nunique
    })
    df_daily.columns = ['CA_total', 'CA_moyen', 'nb_transactions', 'nb_magasins']
    
    # Fusion avec météo (moyenne pondérée par ville)
    df_weather = df_weather.groupby('Date').agg({
        'Température': 'mean',
        'Pluie_mm': 'max'
    })
    df_merged = pd.merge(df_daily, df_weather, left_index=True, right_index=True, how='left')
    
    # Imputation intelligente
    df_merged['Température'] = df_merged['Température'].interpolate()
    df_merged['Pluie_mm'] = df_merged['Pluie_mm'].fillna(0)
    
    # Features temporelles
    df_merged['jour_semaine'] = df_merged.index.dayofweek
    df_merged['weekend'] = (df_merged['jour_semaine'] >= 5).astype(int)
    
    # Jours fériés
    fr_holidays = holidays.France(years=2024)
    df_merged['ferie'] = df_merged.index.map(lambda x: x in fr_holidays).astype(int)
    
    # Promotions
    promo_periods = [
        (row['Date_début'], row['Date_fin']) 
        for _, row in df_marketing.iterrows()
    ]
    df_merged['promo'] = 0
    for start, end in promo_periods:
        df_merged.loc[start:end, 'promo'] = 1
    
    return df_merged.reset_index()

# ---- Interface Streamlit ----
def main():
    data = load_data()
    
    # ---- Sidebar avancée ----
    with st.sidebar:
        st.header("⚙️ Paramètres Avancés")
        
        model_config = {
            'seasonality_mode': st.selectbox(
                "Saisonnalité",
                ['additive', 'multiplicative'],
                index=1,
                help="Additive pour variations constantes, multiplicative pour variations proportionnelles"
            ),
            'changepoint_scale': st.slider(
                "Sensibilité aux changements", 
                0.01, 0.5, 0.1, 0.01,
                help="Contrôle la flexibilité de la courbe de tendance"
            ),
            'holidays_scale': st.slider(
                "Impact des jours fériés",
                1.0, 20.0, 10.0, 0.5
            )
        }
        
        st.divider()
        st.markdown("**Options d'affichage**")
        show_components = st.checkbox("Afficher les composantes", True)
        show_metrics = st.checkbox("Afficher les métriques", True)
    
    # ---- Préparation des données ----
    df = enhance_data(data['ventes_enhanced'], data['meteo_locale'], data['campagnes_marketing'])
    
    # ---- Onglets d'analyse ----
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Analyse", "📤 Export"])
    
    with tab1:
        # KPI Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("CA Total", f"{df['CA_total'].sum()/1000:.1f}K€")
        col2.metric("Magasins Moyens/Jour", f"{df['nb_magasins'].mean():.0f}")
        col3.metric("Température Moyenne", f"{df['Température'].mean():.1f}°C")
        
        # Graphiques interactifs
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['CA_total'], 
            name="CA Réel", line=dict(color='#1f77b4'))
        )
        fig.update_layout(
            title="Performance Commerciale Journalière",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cartographie des corrélations
        st.subheader("Matrice de Corrélation")
        corr_matrix = df[['CA_total', 'Température', 'Pluie_mm', 'weekend', 'ferie', 'promo']].corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
    
    with tab2:
        # ---- Modélisation ----
        st.subheader("Modèle Prédictif Prophet")
        
        # Préparation pour Prophet
        prophet_df = df[['Date', 'CA_total', 'Température', 'weekend', 'ferie', 'promo']]
        prophet_df = prophet_df.rename(columns={
            'Date': 'ds',
            'CA_total': 'y'
        })
        
        # Entraînement
        with st.spinner('Optimisation du modèle en cours...'):
            model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=False,
                seasonality_mode=model_config['seasonality_mode'],
                changepoint_prior_scale=model_config['changepoint_scale'],
                holidays_prior_scale=model_config['holidays_scale']
            )
            
            # Ajout des régresseurs
            for reg in ['Température', 'weekend', 'ferie', 'promo']:
                model.add_regressor(reg)
            
            model.fit(prophet_df)
            
            # Prévisions
            future = model.make_future_dataframe(periods=30)
            future = future.merge(
                prophet_df[['ds', 'Température', 'weekend', 'ferie', 'promo']],
                on='ds',
                how='left'
            ).fillna(method='ffill')
            
            forecast = model.predict(future)
        
        # Visualisation
        st.plotly_chart(
            plot_plotly(model, forecast, xlabel='Date', ylabel='CA (€)'), 
            use_container_width=True
        )
        
        if show_components:
            st.plotly_chart(
                plot_components_plotly(model, forecast),
                use_container_width=True
            )
        
        if show_metrics:
            st.subheader("Validation Croisée")
            df_cv = cross_validation(
                model,
                initial='45 days',
                period='15 days',
                horizon='30 days',
                parallel="processes"
            )
            st.dataframe(
                performance_metrics(df_cv).style.highlight_min(
                    subset=['mape', 'rmse'], color='#fffd75'
                )
            )
    
    with tab3:
        # ---- Export des résultats ----
        st.subheader("Téléchargement des Prévisions")
        
        # Formatage des résultats
        forecast_df = forecast.set_index('ds')[
            ['yhat', 'yhat_lower', 'yhat_upper']
        ].join(df.set_index('Date')['CA_total'])
        
        # Options d'export
        export_format = st.radio(
            "Format d'export",
            ['CSV', 'Excel', 'JSON'],
            horizontal=True
        )
        
        if export_format == 'CSV':
            csv = forecast_df.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger en CSV",
                data=csv,
                file_name='quickmart_forecast.csv',
                mime='text/csv'
            )
        elif export_format == 'Excel':
            excel = forecast_df.reset_index().to_excel("forecast.xlsx", index=False)
            with open("forecast.xlsx", "rb") as f:
                st.download_button(
                    label="📥 Télécharger en Excel",
                    data=f,
                    file_name='quickmart_forecast.xlsx'
                )
        else:
            json = forecast_df.reset_index().to_json(orient='records')
            st.download_button(
                label="📥 Télécharger en JSON",
                data=json,
                file_name='quickmart_forecast.json',
                mime='application/json'
            )
        
        # Code embarqué
        st.divider()
        st.subheader("Intégration API")
        st.code("""
import requests
API_URL = "https://api.quickmart.com/forecast"
params = {
    'start_date': '2024-04-01',
    'end_date': '2024-04-30'
}
response = requests.get(API_URL, params=params)
forecast_data = response.json()
        """)

if __name__ == '__main__':
    main()