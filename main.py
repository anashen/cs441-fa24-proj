import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from skorch import NeuralNetClassifier
import joblib
import os

@st.cache_data
def load_data():
    return pd.read_csv('data/diagnosed_cbc_data.csv', engine="pyarrow")

class ADNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(ADNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[2])
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(hidden_dims[2], output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)
        
        x = self.fc5(x)
        return x


def train_model():
    df = load_data()
    df.drop(['LYMp', 'NEUTp', 'LYMn', 'NEUTn'], axis=1, inplace=True)

    df["Diagnosis_Encoded"] = df["Diagnosis"].astype("category").cat.codes
    diagnosis_label_dict = dict(enumerate(df["Diagnosis"].astype("category").cat.categories))
    print(diagnosis_label_dict)

    X = df.drop(["Diagnosis", "Diagnosis_Encoded"], axis=1).values
    y = df["Diagnosis_Encoded"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    model_nn = NeuralNetClassifier(
        ADNN,
        module__input_dim=X.shape[1],
        module__hidden_dims=[256, 128, 64],
        module__output_dim=len(diagnosis_label_dict),
        max_epochs=100,
        lr=0.001,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        batch_size=64,
        iterator_train__shuffle=True
    )

    model_dt = DecisionTreeClassifier(random_state=7)

    ensemble_model = VotingClassifier(estimators=[
        ('nn', model_nn),
        ('dt', model_dt)
    ], voting='hard')

    ensemble_model.fit(X_train, y_train)

    accuracy_em = ensemble_model.score(X_test, y_test)
    print(f"~~~~ Model Accuracy: {accuracy_em * 100:.2f}%")

    # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    # cv_scores_em = cross_val_score(ensemble_model, X.astype(np.float32), y.astype(np.int64), cv=cv, scoring='accuracy')

    # print(f"~~~~ CV Mean Model Accuracy: {cv_scores_em.mean() * 100:.2f}%")

    joblib.dump(ensemble_model, 'model_ensemble.pkl')
    joblib.dump(scaler, 'model_scaler.pkl')

    return ensemble_model, scaler, diagnosis_label_dict

def main():
    st.title("Anemia Diagnostic Tool")

    if not os.path.exists('model_ensemble.pkl'):
        with st.spinner('Training model...'):
            ensemble_model, scaler, diagnosis_mapping = train_model()
        st.write("Model trained successfully!")
    else:
        with st.spinner('Loading model...'):
            ensemble_model = joblib.load('model_ensemble.pkl')
            scaler = joblib.load('model_scaler.pkl')
            df = load_data()
            diagnosis_mapping = dict(enumerate(df["Diagnosis"].astype("category").cat.categories))
        st.write("Model loaded successfully!")

    option = st.radio(
        '# Select method of data input:',
        ['CSV upload', 'Manual entry']
    )

    if option == 'CSV upload':
        st.write("### Upload patient data for prediction:")
        uploaded_file = st.file_uploader("Choose a file...")
        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file, header=None)

            udf_tensor = torch.tensor(uploaded_df.to_numpy().astype(np.float32), dtype=torch.float32)
            udf_tensor_scaled = scaler.transform(udf_tensor.numpy())
            udf_tensor_scaled = torch.tensor(udf_tensor_scaled, dtype=torch.float32)

            outputs = ensemble_model.predict(udf_tensor_scaled)
            predicted_diagnoses = [diagnosis_mapping[pred] for pred in outputs]
            
            patient_predictions = pd.DataFrame(predicted_diagnoses)
            patient_predictions_csv = patient_predictions.to_csv(index=False).encode('utf-8')

            st.download_button(
                "Download predicted diagnoses",
                patient_predictions_csv,
                "predicted.csv",
                "text/csv",
                key='download-csv'
            )
            
            patient_predictions.columns = ['Diagnosis']
            fig = px.histogram(patient_predictions, x='Diagnosis')
            st.plotly_chart(fig)

            # # (Test uploaded) Verify that the diagnoses are accurate
            # uploaded_df_y = pd.read_csv("data/ex_patient_cbc_diagnoses.csv", header=None)
            # uploaded_df_y.columns = ['(True) Diagnosis Category']
            # fig2 = px.histogram(uploaded_df_y, x='(True) Diagnosis Category')
            # st.plotly_chart(fig2)
            
    elif option == 'Manual entry':
        st.write("### Enter patient data for prediction:")
        with st.form('form'):
            col1, col2, col3 = st.columns(3)
            wbc = col1.number_input("WBC", min_value=0., max_value=40., step=0.01, format="%0.2f", help="Count of white blood cells, x 10^9/l")
            rbc = col1.number_input("RBC", min_value=0., max_value=40., step=0.01, format="%0.2f", help="Count of red blood cells, x 10^12/l")
            hgb = col1.number_input("HGB", min_value=0., max_value=50., step=0.01, format="%0.2f", help="Amount of hemoglobin, g/l")
            hct = col2.number_input("HCT", min_value=0., max_value=100., step=0.01, format="%0.2f", help="Percentage of red blood cells (hematocrit)")
            mcv = col2.number_input("MCV", min_value=0., max_value=200., step=0.01, format="%0.2f", help="Average volume of a single red blood cell, fl")
            mch = col2.number_input("MCH", min_value=0., max_value=500., step=0.01, format="%0.2f", help="Average amount of hemoglobin per red blood cell, pg")
            mchc = col3.number_input("MCHC", min_value=0., max_value=50., step=0.01, format="%0.2f", help="Average concentration of hemoglobin in red blood cells, g/l")
            pltl = col3.number_input("PLT", min_value=0., max_value=500., step=0.01, format="%0.2f", help="Count of platelets in the blood, x 10^9/l")
            pdw = col3.number_input("PDW", min_value=0., max_value=50., step=0.01, format="%0.2f", help="Percentage variability in platelet width distribution")
            pct = col3.number_input("PCT", min_value=0., max_value=5., step=0.01, format="%0.2f", help="Level of procalcitonin (sepsis biomarker) in the blood, µg/L")
            submit = st.form_submit_button('Submit')

        if submit:
            input_data = np.array([[wbc, rbc, hgb, hct, mcv, mch, mchc, pltl, pdw, pct]])

            input_data_scaled = scaler.transform(input_data)
            input_data_scaled = input_data_scaled.astype(np.float32)

            predicted_class = ensemble_model.predict(input_data_scaled)[0]
            predicted_diagnosis = diagnosis_mapping[predicted_class]
            st.write(f"#### Predicted Diagnosis: {predicted_diagnosis}")

if __name__ == "__main__":
    main()
