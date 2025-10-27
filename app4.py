import streamlit as st
import numpy as np
import joblib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Besoin Crèche - Dimensionnement",
    page_icon="",
    layout="centered"
)

#Style de la page 
st.markdown("""
<style>
body {
    background-color: #F9FAFB;
    font-family: "Helvetica", sans-serif;
}
h1, h2, h3 {
    color: #003366;
}
div.stButton > button {
    background-color: #003366;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 1.1em;
}
div.stButton > button:hover {
    background-color: #004B8C;
}
.result-card {
    background-color: #DCE6F2;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}
.top-banner {
    background-color: #003366;
    color: white;
    padding: 12px;
    text-align: center;
    border-radius: 0 0 15px 15px;
}
.footer {
    text-align:center;
    color:gray;
    font-size:13px;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# Bande blue et et logo
st.markdown("""
        <div style="background-color:#02378E;padding:10px;border-radius:10px;text-align:center;">
            
        </div>
    """, unsafe_allow_html=True)
st.write(" ")
# Image d'accueil
st.image(
    "image.png",
    use_container_width=True,
    #caption="Outil dimensionnement des besoins en crèche"
)

# Titre et descrition de l'outils
st.markdown("""
<h1 style='text-align:center;'> Outil de dimensionnement des besoins en crèche</h1>
<p style='text-align:center; color:gray; font-size:17px;'>
Obtenez une estimation du nombre de places à réserver en crèche selon le profil de votre entreprise.
Un outil simple et intuitif pour orienter vos choix.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------ Choix du type d'entreprise ------------------
st.markdown("### Choisissez le type d’entreprise")
type_entreprise = st.selectbox(
    "Sélectionnez le type d’entreprise :",
    ["PME", "Grand Compte"]
)

# ------------------ Chargement du modèle de régression quantile ------------------
model_quantile = joblib.load("modele_quantile.pkl")

# ------------------ Champs à saisir ------------------
if type_entreprise:
    st.markdown("---")
    st.markdown("#### Entrez les caractéristiques de votre entreprise")

    col1, col2 = st.columns(2)
    with col1:
        Nb_Sal = st.number_input(" Nombre  de salariés", min_value=0, value=250)
        Taux_F = st.number_input(" Taux de féminisation (%)", min_value=0.0, max_value=100.0, value=43.0)
    with col2:
        Part_JE = st.number_input(" Part de jeunes enfants (%)", min_value=0.0, max_value=100.0, value=10.0)
        Prct_TpsP = st.number_input("Le temps partiels (%)", min_value=0.0, max_value=100.0, value=8.0)
        Prct_Moins40 = st.number_input("Part des moins de 40 (%)", min_value=0.0, max_value=100.0, value=8.2)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Estimer le besoin en places", use_container_width=True)

    # ------------------ Prédiction du besoin ------------------
    if predict_btn:
        # Conversion des pourcentages en proportions
        #Taux_F = Taux_F / 100
        #Part_JE = Part_JE / 100
        #Prct_TpsP = Prct_TpsP / 100
        seg_val = 1 if type_entreprise == "Grand Compte" else 0

        # Variables explicatives selon ta formule
        #Nbs_SE = np.log1p(Nb_Sal) * seg_val
        #F_TpsP = Taux_F * Prct_TpsP
        #JE_TpsP = Prct_TpsP * Part_JE

        # DataFrame avec les bons noms de variables
        X_df = pd.DataFrame({
            "Nb_Sal": [Nb_Sal],
            "Segment_Entreprise":[seg_val],
            "Taux_F": [Taux_F],
            "Part_JE": [Part_JE],
            "Prct_TpsP": [Prct_TpsP],
            "Prct_Moins40":[Prct_Moins40]
        })

        # Application du modèle de régression quantile
        X_sm = sm.add_constant(X_df,has_constant='add')
        prediction = model_quantile.get_prediction(X_sm)
        pred_summary = prediction.summary_frame(alpha=0.05)

        # Détection automatique du bon nom de colonne (OLS ou GLM)
        if "mean" in pred_summary.columns:
            pred_col = "mean"
        elif "predicted_mean" in pred_summary.columns:
            pred_col = "predicted_mean"
        else:
            pred_col = pred_summary.columns[0]

        # Récupération de la prédiction et des intervalles de confiance
        pred_q = pred_summary[pred_col][0]
        pred_low = pred_summary["mean_ci_lower"][0]
        pred_high = pred_summary["mean_ci_upper"][0]

        avg_cap = round(pred_q)
        min_cap = round(pred_low)
        max_cap = round(pred_high)

        # ------------------ Affichage du résultat ------------------
        st.markdown("---")
        st.markdown("#### Estimation du besoin en crèche")

        st.markdown(f"""
        <div class='result-card'>
            <h2 style='color:#003366;'>Besoin estimé : <b>{avg_cap} places</b></h2>
            <p style='font-size:17px; color:#1E3A5F;'>
           Pour des entreprises au profil similaire, le besoin moyen est d’environ {avg_cap} places,
            avec une estimation variant entre {min_cap} et {max_cap} places.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(avg_cap / 300, 1.0))

        st.markdown("""
        <div style='text-align:center; color:gray; margin-top:10px;'>
            <em>Cette estimation repose sur les paramètres saisis.</em><br>
            Modifiez les valeurs pour simuler d'autres profils d'entreprise.
        </div>
        """, unsafe_allow_html=True)

## fin de page
st.markdown("""
<div class='footer'>
Réalisé par Thierno | Outil de dimensionnement © 2025
</div>
""", unsafe_allow_html=True)
