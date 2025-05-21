import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import joblib
import numpy as np
import pandas as pd
import base64
import random
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

diabetes_df = pd.read_csv("datasets/diabetes_prediction_dataset.csv", low_memory=False)
diabetes_df['smoking_history'] = diabetes_df['smoking_history'].replace('No Info', np.nan)
diabetes_df.dropna(inplace=True)
diabetes_df['age'] = pd.to_numeric(diabetes_df['age'], errors='coerce') 
diabetes_df = diabetes_df[diabetes_df['age'] % 1 == 0] 
diabetes_model = joblib.load("./models/diabetes_prediction_model.sav")

diabetes = dbc.Container(
    dbc.Card(
        dbc.CardBody([
            html.H2("Diabetes Prediction", className="text-center mt-4 mb-4", style={"fontWeight": "bold"}),
            dbc.Row([
                dbc.Col(dbc.Button("Fill Random Data", id="fill-random-btn", color="secondary", className="mb-4"), width="auto")
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Input(id="name_diabetes", type="text", placeholder="Name", className="form-control"), md=4),
                dbc.Col(dcc.Dropdown(id="sex_diabetes", options = [{'label':'Female', 'value':0}, {'label':'Male', 'value':1}], placeholder='Sex'), md=4),
                dbc.Col(dcc.Input(id="age", type="number", step=1, placeholder="Age", className="form-control"), md=4),
            ], className="mb-1"),
            
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="hypertension", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Have Hypertension"), md=4),
                dbc.Col(dcc.Dropdown(id="heart_disease", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Had Heart Disease"), md=4),
                dbc.Col(dcc.Dropdown(id="smoking_history", options=[
                    {"label": "Never", "value": 0},
                    {"label": "Current", "value": 1},
                    {"label": "Former", "value": 2},
                    {"label": "Ever", "value": 3},
                    {"label": "Not Current", "value": 4},
                ], placeholder="Smoking History"), md=4),
            ], className="mb-1"),
                
            dbc.Row([
                dbc.Col(dcc.Input(id="bmi", type="number", step=0.01, placeholder="BMI", className="form-control"), md=4),
                dbc.Col(dcc.Input(id="HbA1c_level", type="number", step=0.1, placeholder="HbA1C Level", className="form-control"), md=4),
                dbc.Col(dcc.Input(id="blood_glucose_level", type="number", step=1, placeholder="Blood Glucose Level", className="form-control"), md=4),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dbc.Button("Get Prediction", id="predict-btn", color="primary", className="mt-3"), md=6),
                dbc.Col(dbc.Button("Download Report", id="download-btn", color="success", className="mt-3"), md=6),
            ], className="mb-1"),

            html.Div(id="prediction-result-diabetes", className="mt-4 text-center"),
            dcc.Download(id="download-pdf-diabetes")
        ], className="green-gradient"),
        className="shadow p-4 green-gradient",
        style={"borderRadius": "15px", "maxWidth": "1000px", "marginTop": "5%"}
    ),
    fluid=True,
    className="d-flex justify-content-center align-items-center"
)

@dash.callback(
    Output("prediction-result-diabetes", "children"),
    Input("predict-btn", "n_clicks"),
    [
        State("name_diabetes", "value"),
        State("sex_diabetes", "value"),
        State("age", "value"),
        State("hypertension", "value"),
        State("heart_disease", "value"),
        State("smoking_history", "value"),
        State("bmi", "value"),
        State("HbA1c_level", "value"),
        State("blood_glucose_level", "value"),
    ]
)

def predict_diabetes(n_clicks, *values):

    name_diabetes, sex_diabetes, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level = values
    if not n_clicks:
        return ""

    # Check for missing inputs
    if None in [name_diabetes, sex_diabetes, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]:
        return html.Div("⚠️ Please fill all fields before predicting.", className="alert alert-warning")

    # Create input array
    features = np.array([[sex_diabetes, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])

    # Predict using preloaded model
    prediction = diabetes_model.predict(features)

    # Return result
    if prediction[0] == 1:
        return html.Div(f"🩺 {name_diabetes}, you are likely to have **Diabetes**.", className="alert alert-danger")
    else:
        return html.Div(f"✅ {name_diabetes}, you are **not likely** to have Diabetes.", className="alert alert-success")
    return ''

@dash.callback(
    Output("download-pdf-diabetes", "data"),
    Input("download-btn", "n_clicks"),
    [
        State("name_diabetes", "value"),
        State("sex_diabetes", "value"),
        State("age", "value"),
        State("hypertension", "value"),
        State("heart_disease", "value"),
        State("smoking_history", "value"),
        State("bmi", "value"),
        State("HbA1c_level", "value"),
        State("blood_glucose_level", "value"),
        State("prediction-result-diabetes", "children")
    ],
    prevent_initial_call=True
)
def download_diabetes_report(n_clicks, *values):
    name_diabetes, sex_diabetes, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, prediction_result = values
    if not n_clicks or None in [name_diabetes, sex_diabetes, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, prediction_result]:
        return None

    # Handle Dash component in prediction_result
    if isinstance(prediction_result, dict) and "props" in prediction_result:
        prediction_text = prediction_result["props"].get("children", "Prediction not available")
    else:
        prediction_text = str(prediction_result)

    labels_values = [
        ("Name", name_diabetes),
        ("Sex", "Male" if sex_diabetes == 1 else "Female"),
        ("Age", age),
        ("Hypertension", "Yes" if hypertension == 1 else "No"),
        ("Heart Disease", "Yes" if heart_disease == 1 else "No"),
        ("Smoking History", {0: "Never",
                            1: "Current",
                            2: "Former",
                            3: "Ever",
                            4: "Not Current"}.get(smoking_history, "Unknown")),
        ("BMI", bmi),
        ("HbA1c Level", HbA1c_level),
        ("Blood Glucose Level", blood_glucose_level),
    ]

    # Generate PDF
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)

    y = 750
    pdf.drawString(100, y, "Diabetes Prediction Report")
    y -= 10
    pdf.line(100, y, 500, y)
    y -= 30

    for label, value in labels_values:
        pdf.drawString(100, y, f"{label}: {value}")
        y -= 20
        if y < 100:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = 750

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(100, y - 10, f"Prediction Result: {prediction_text}")

    pdf.save()
    buffer.seek(0)

    return dcc.send_bytes(buffer.getvalue(), filename=f"{name_diabetes}_Diabetes_Prediction_Report.pdf")

@dash.callback(
    Output("name_diabetes", "value"),
    Output("sex_diabetes", "value"),
    Output("age", "value"),
    Output("hypertension", "value"),
    Output("heart_disease", "value"),
    Output("smoking_history", "value"),
    Output("bmi", "value"),
    Output("HbA1c_level", "value"),
    Output("blood_glucose_level", "value"),
    Input("fill-random-btn", "n_clicks"),
    prevent_initial_call=True
)
def fill_from_dataset(n_clicks):
    # Randomly sample one row
    sample = diabetes_df.sample(1).iloc[0]

    name_diabetes = random.choice(["Alex", "Sam", "Jamie", "Taylor", "Jordan", "Riya", "Karan", "Aisha"])
    
    # Prepare values
    sex_diabetes_map = {'Male': 1, "Female": 0}
    sex_diabetes = sex_diabetes_map.get(str(sample['gender']), 0)
    age = int(sample["age"])
    hypertension = int(sample["hypertension"])
    heart_disease = int(sample["heart_disease"])
    smoking_map = {
        "never": 0, "current": 1, "former": 2, "ever": 3, "not current": 4
    }
    smoking_history = smoking_map.get(str(sample["smoking_history"]).lower(), 0)
    bmi = round(float(sample["bmi"]), 2)
    HbA1c_level = round(float(sample["HbA1c_level"]), 1)
    blood_glucose_level = int(sample["blood_glucose_level"])

    return name_diabetes, sex_diabetes, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level



