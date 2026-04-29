import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

csv_file = './penguins.csv'
df = pd.read_csv(csv_file)

df.dropna(inplace=True)

df['sex'] = df['sex'].str.strip().str.capitalize()
df['island'] = df['island'].str.strip().str.capitalize()
df['species'] = df['species'].str.strip().str.capitalize()

df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen': 2})

df.dropna(inplace=True)

X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']
score = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

svc_model = SVC(kernel='linear')
svc_model.fit(X_train, y_train)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

rf_clf = RandomForestClassifier(n_jobs=-1)
rf_clf.fit(X_train, y_train)

species_images = {
    "Adelie": "https://cdn.britannica.com/77/81277-050-2A6A35B2/Adelie-penguin.jpg",
    "Chinstrap": "https://www.asoc.org/wp-content/uploads/2024/02/cropped-dancing-Chinstrap-penguin-787x650.png",
    "Gentoo": "https://i.pinimg.com/736x/47/92/ba/4792ba991ad3cb03f74fcfef101fcb7c.jpg"
}

@st.cache_data()
def prediction(_model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    species = _model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
    return species

st.title("Penguin ML project")
st.sidebar.title("Penguin Predictor")

bill_length_mm = st.sidebar.slider('bill_length_mm', float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
bill_depth_mm = st.sidebar.slider('bill_depth_mm', float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
flipper_length_mm = st.sidebar.slider('flipper_length_mm', float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
body_mass_g = st.sidebar.slider('body_mass_g', float(df["body_mass_g"].min()), float(df["body_mass_g"].max()))
sex = st.sidebar.radio("Sex", ('Male', 'Female'))
island = st.sidebar.radio('island', ('Biscoe', 'Dream', 'Torgersen'))
model = st.sidebar.radio('Select Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

sex = 0 if sex == 'Male' else 1

if island == 'Biscoe':
    island = 0
elif island == 'Dream':
    island = 1
else:
    island = 2

if st.sidebar.button('Predict'):
    if model == 'Support Vector Machine':
        species = prediction(svc_model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
        score = svc_model.score(X_train, y_train)
    elif model == 'Logistic Regression':
        species = prediction(log_reg, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
        score = log_reg.score(X_train, y_train)
    elif model == 'Random Forest Classifier':
        species = prediction(rf_clf, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
        score = rf_clf.score(X_train, y_train)

    species_name = ["Adelie", "Chinstrap", "Gentoo"][int(species[0])]

    st.markdown("""
    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
        <h2 style="color: #2e8b57; text-align: center;">Species: {}</h2>
        <div style="text-align: center;">
            <img src="{}" alt="{}" style="width: 300px; border-radius: 10px;"/>
        </div>
        <p style="font-size:20.0px; color:#000;">Accuracy score of this model is: <p style="font-size:25px; color:#000;">{}</p></p>
    </div>
    """.format(species_name, species_images[species_name], species_name, score), unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background-color: #ffcccb; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h3 style="color: #d32f2f; text-align: center;">Please select the appropriate settings and click Predict</h3>
    </div>
    """, unsafe_allow_html=True)
