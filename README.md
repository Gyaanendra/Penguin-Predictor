# 🐧 Penguin Species Predictor

A machine learning web app built with **Streamlit** that predicts the species of a penguin based on physical measurements. Choose from three classifiers and get instant predictions with accuracy scores.

🌐 **Live Demo:** [https://penguin-gp.streamlit.app/](https://penguin-gp.streamlit.app/)

---

## 📌 Overview

This app uses the **Palmer Penguins dataset** to train three ML models and allows users to interactively input penguin measurements to predict whether a penguin is:

- 🟠 **Adelie**
- 🔵 **Chinstrap**
- 🟢 **Gentoo**

---

## 🚀 Features

- Interactive sidebar with sliders and radio buttons for input
- Three classifier options: SVM, Logistic Regression, and Random Forest
- Displays the predicted species with an image
- Shows the training accuracy score of the selected model

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web UI framework |
| Scikit-learn | ML models |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Visualization support |

---

## 📁 Project Structure

```
├── penguin_app.py      # Main Streamlit app
├── penguins.csv        # Dataset (Palmer Penguins)
└── README.md           # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/penguin-predictor.git
cd penguin-predictor
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn streamlit scikit-learn
```

### 3. Add the dataset

Place the `penguins.csv` file in the same directory as `penguin_app.py`.  
You can download it from [palmerpenguins](https://github.com/allisonhorst/palmerpenguins) or use the CSV from the seaborn datasets.

### 4. Run the app

```bash
streamlit run penguin_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 ML Models

| Model | Description |
|---|---|
| **Support Vector Machine (SVC)** | Linear kernel SVM |
| **Logistic Regression** | Multinomial logistic regression |
| **Random Forest Classifier** | Ensemble of decision trees (`n_jobs=-1`) |

All models are trained on a **67/33 train-test split** with `random_state=42`.

---

## 📊 Input Features

| Feature | Type | Description |
|---|---|---|
| `island` | Categorical | Biscoe, Dream, or Torgersen |
| `bill_length_mm` | Numeric | Length of the bill in mm |
| `bill_depth_mm` | Numeric | Depth of the bill in mm |
| `flipper_length_mm` | Numeric | Flipper length in mm |
| `body_mass_g` | Numeric | Body mass in grams |
| `sex` | Categorical | Male or Female |

---

## 🖼️ App Preview

Use the **sidebar** to:
1. Adjust penguin measurements using sliders
2. Select the penguin's sex and island
3. Choose a classifier
4. Click **Predict** to see the result
