import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# dans le requirements.txt
# ne pas installer time (ou autre module)
# qui est un module standard à Python

# local run
# streamlit run ds_et_ml.py

##################################################
# Configurer la page
# wide, centered
# auto or expanded
#menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
#            'Report a bug': "https://www.extremelycoolapp.com/bug",
#            'About': "# This is a header. This is an *extremely* cool app!"}
st.set_page_config(page_title="Tableau de bord",
                   page_icon="img/favicon-32x32b.png",
                   layout="centered",
                   initial_sidebar_state="expanded")

# Injecter du CSS
# https://html-color.codes/
# default, unsafe_allow_html=False
st.markdown(
    """
    <style>
     .main {
     background-color: #3b444b;
     }
    </style>
    """,
    unsafe_allow_html=True
)

##################################################

# Créer des conteneurs
# https://www.youtube.com/watch?v=CSv2TBA9_2E&list=PLM8lYG2MzHmRpyrk9_j9FW0HiMwD9jSl5&index=3
# Part 1 à 4
siteHeader = st.container()
dataExploration = st.container()
newFeatures = st.container()
modelTraining = st.container()

# Utiliser le conteneur
with siteHeader:
    st.title('Tableau de DS et ML')

    # Injecter du HTML dans le Markdown
    # default, unsafe_allow_html=False
    st.markdown(
        """
        <b>Un tableau de bord intermédiaire</b>
        """,
        unsafe_allow_html=True
    )

# Utiliser le conteneur
with dataExploration:
    st.header('Données')
    
    st.markdown('Avec le jeu de données: NYC Taxi.')

    # Fonction d'importation
    # sans réimporter avec la mémoire
    # https://learn.microsoft.com/en-us/azure/open-datasets/dataset-taxi-yellow?tabs=azureml-opendatasets
    @st.cache
    def get_data():
        taxi_data = pd.read_parquet('data/yellow_tripdata_2022-01.parquet')
        return taxi_data
    
    taxi_data = get_data()
    st.write(taxi_data.head(3))
    
    st.subheader('Tous les features')
    
    st.markdown(f'Voyons la liste de features: {list(taxi_data.columns)}')
    st.write(f'Voyons la liste de features: {list(taxi_data.columns)}')
    
    st.subheader('Un feature')
    
    st.markdown('Voyons la distribution de `PULocationID`.')
    distribution_pickup = pd.DataFrame(taxi_data['PULocationID'].value_counts())
    st.bar_chart(distribution_pickup)

# Utiliser le conteneur
with newFeatures:
    st.header('Ajout de nouveaux features')
    
    # Utiliser le Markdown multilignes
    st.markdown(
        """
        - **Nouveau feature:** explication
        - **Nouveau feature:** explication
        """
    )

# Utiliser le conteneur
with modelTraining:
    st.header('Modèle de Random Forests')

    # Créer 2 colonnes
    selection_col, display_col = st.columns(2)
    
    # Colonne selection_col
    
    selection_col.subheader('Sélectionnons les hyperparamètre.')
    
    # Insérer un widget
    max_depth = selection_col.slider('Profondeur des arbres ou `max_depth=`',
                                     min_value=10,
                                     max_value=100,
                                     value=10,
                                     step=10)
    # Insérer un widget
    number_of_estim = selection_col.selectbox('Nombre d\'arbres ou `n_estimators=`',
                                              options=[25, 50, 75, 100, 125, 150, 200, 'No limit'],
                                              index=0)
    
    selection_col.subheader('Sélectionnons un feature')
    
    selection_col.markdown('Voici une liste de features: ')
    selection_col.write(taxi_data.columns)
    # Insérer un widget
    input_feature = selection_col.text_input('Remplacer ou ajouter un feature au modèle:',
                                             'PULocationID')
        
    # Colonne display_col

    display_col.subheader('Évaluons le modèle')

    
    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]
    
    display_col.markdown(f'`y = trip_distance`')
    display_col.markdown(f'`X = {input_feature}`')
    
    if number_of_estim is 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth,
                                     n_estimators=number_of_estim)

    regr.fit(X, y)
    prediction = regr.predict(y)

    display_col.markdown(f'**Mean Absolute Error:** {round(mean_absolute_error(y, prediction), 3)}')
    display_col.markdown(f'**Mean Squared Error:** {round(mean_squared_error(y, prediction), 3)}')
    