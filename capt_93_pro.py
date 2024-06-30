# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'penguin_app.py'.

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

species_images = {
    "Adelie": "https://cdn.britannica.com/77/81277-050-2A6A35B2/Adelie-penguin.jpg",
    "Chinstrap": "https://www.asoc.org/wp-content/uploads/2024/02/cropped-dancing-Chinstrap-penguin-787x650.png",
    "Gentoo": "https://i.pinimg.com/736x/47/92/ba/4792ba991ad3cb03f74fcfef101fcb7c.jpg"
}

@st.cache_data()
def prediction(_model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
  species = _model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
  return species

# app Design
st.title("Penguin ML project ")
st.sidebar.title("Penguin Predictor")
bill_length_mm = st.sidebar.slider('bill_length_mm', float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
bill_depth_mm = st.sidebar.slider('bill_depth_mm', float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
flipper_length_mm = st.sidebar.slider('flipper_length_mm', float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
body_mass_g = st.sidebar.slider('body_mass_g',float(df["body_mass_g"].min()), float(df["body_mass_g"].max()))
sex = st.sidebar.radio("Sex",('Male', 'Female'))
island = st.sidebar.radio('island', ('Biscoe', 'Dream', 'Torgersen'))
model = st.sidebar.radio('Select Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))


if sex == 'Male':
  sex = 0
else:
  sex = 1

if island == 'Biscoe':
  island = 0
elif island == 'Dream':
  island = 1
else:
  island = 2
                     
if st.sidebar.button('Predict'):
  if model == 'Support Vector Machine':
     species = prediction(svc_model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
  elif model == 'Logistic Regression':
     species = prediction(log_reg, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
  elif model == 'Random Forest Classifier':
     species = prediction(rf_clf, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
  else:
    st.write("Please select a model")
    
  if species == 0:
      species = "Adelie"
  elif species == 1:
      species = "Chinstrap"
  else:
      species = "Gentoo"
            
  st.markdown("""
    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
        <h2 style="color: #2e8b57; text-align: center;">Species: {}</h2>
        <div style="text-align: center;">
            <img src="{}" alt="{}" style="width: 300px; border-radius: 10px;"/>
        </div>
    </div>
    """.format(species, species_images[species], species), unsafe_allow_html=True)
else:
   st.markdown("""
    <div style="background-color: #ffcccb; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h3 style="color: #d32f2f; text-align: center;">Please select the appropriate settings and click Predict</h3>
    </div>
    """, unsafe_allow_html=True)
