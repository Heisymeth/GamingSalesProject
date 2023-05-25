import streamlit as st

st.title('Visualisation PowerBI')

st.markdown('Analyse du DataFrame d\'origine et du DataFrame final')

# Personnalisation CSS pour agrandir l'iframe
css = """
    <style>
    iframe {
        width: 200%;
        height: 700px;
    }
    </style>
    """

# Affichage du CSS personnalisé
st.markdown(css, unsafe_allow_html=True)

# Affichage de l'iframe contenant le rapport Power BI
st.components.v1.iframe("https://app.powerbi.com/reportEmbed?reportId=a3a4cd49-8798-4a3e-b348-1368b89c7192&autoAuth=true&embeddedDemo=true")

