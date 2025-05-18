import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import joblib
import pandas as pd
import numpy as np
import base64
import io
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

heart_df = pd.read_csv("datasets/heart_2022.csv", low_memory=False)
heart_df.dropna(inplace=True)
heart_disease_model = joblib.load("./models/heart_disease_model.sav")


heart = dbc.Container(
    dbc.Card(
        dbc.CardBody([
            html.H2("Heart Disease Prediction", className="text-center mt-4 mb-4", style={"fontWeight": "bold"}),

            dbc.Row([
                dbc.Col(dbc.Button("Fill Random Data", id="fill-random-btn", color="secondary", className="mb-4"), width="auto")
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="GeneralHealth", options=[
                    {"label": "Poor", "value": 3},
                    {"label": "Fair", "value": 1},
                    {"label": "Good", "value": 2},
                    {"label": "Very Good", "value": 4},
                    {"label": "Excellent", "value": 0},
                ], placeholder="General Health"), md=6),
                dbc.Col(dcc.Dropdown(id="PhysicalActivities", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Physical Activities"), md=4),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col(dcc.Input(id="SleepHours", type="number", step=1, placeholder="Sleep Hours", className="form-control"), md=4),
                dbc.Col(dcc.Dropdown(id="HadAngina", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Had Angina"), md=4),
                dbc.Col(dcc.Dropdown(id="HadStroke", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Had Stroke"), md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="HadAsthma", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Had Asthma"), md=4),
                dbc.Col(dcc.Dropdown(id="HadSkinCancer", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Had Skin Cancer"), md=4),
                dbc.Col(dcc.Dropdown(id="HadCOPD", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Had COPD"), md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="HadDepressiveDisorder", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Had Depression"), md=4),
                dbc.Col(dcc.Dropdown(id="HadKidneyDisease", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Had Kidney Disease"), md=4),
                dbc.Col(dcc.Dropdown(id="HadArthritis", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Had Arthritis"), md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="HadDiabetes", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Had Diabetes"), md=4),
                dbc.Col(dcc.Dropdown(id="DeafOrHardOfHearing", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Hearing Issue"), md=4),
                dbc.Col(dcc.Dropdown(id="BlindOrVisionDifficulty", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Vision Issue"), md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="DifficultyConcentrating", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Difficulty Concentrating"), md=4),
                dbc.Col(dcc.Dropdown(id="DifficultyWalking", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Difficulty Walking"), md=4),
                dbc.Col(dcc.Dropdown(id="DifficultyDressingBathing", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Dressing/Bathing Difficulty"), md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="DifficultyErrands", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Difficulty Doing Errands"), md=4),
                dbc.Col(dcc.Dropdown(id="ChestScan", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Chest Scan"), md=4),
                dbc.Col(dcc.Dropdown(id="AlcoholDrinkers", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="Alcohol Use"), md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="SmokerStatus", options=[
                    {"label": "Never", "value": 3},
                    {"label": "Former", "value": 2},
                    {"label": "Every Day", "value": 0},
                    {'label': "Some Days", "value": 1}
                ], placeholder="Smoker Status"), md=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="HIVTesting", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="HIV Test"), md=4),
                dbc.Col(dcc.Dropdown(id="HighRiskLastYear", options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], placeholder="High Risk Last Year"), md=4),
            ], className="mb-3"),
                
            dbc.Row([
                dbc.Col(dcc.Input(id="HeightInMeters", type="number", step=0.01, placeholder="Height (m)", className="form-control"), md=4),
                dbc.Col(dcc.Input(id="WeightInKilograms", type="number", step=0.01, placeholder="Weight (kg)", className="form-control"), md=4),
                dbc.Col(dcc.Input(id="BMI", type="number", step=0.01, placeholder="BMI", className="form-control"), md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Input(id="name_heart", type="text", placeholder="Name", className="form-control"), md=6),
                dbc.Col(dcc.Dropdown(id="sex_heart", options = [{'label':'Female', 'value':0}, {'label':'Male', 'value':1}], placeholder='Sex'), md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dbc.Button("Get Prediction", id="predict-btn", color="primary", className="mt-3"), md=6),
                dbc.Col(dbc.Button("Download Report", id="download-btn", color="success", className="mt-3"), md=6),
            ], className="mb-3"),

            html.Div(id="prediction-result-heart", className="mt-4 text-center"),
            dcc.Download(id="download-pdf-heart")
        ], className="blue-gradient"),
        className="shadow p-4 blue-gradient",
        style={"borderRadius": "15px", "maxWidth": "1000px", "marginTop": "5%"}
    ),
    fluid=True,
    className="d-flex flex-column justify-content-center align-items-center"
)

@dash.callback(
    Output("prediction-result-heart", "children"),
    Input("predict-btn", "n_clicks"),
    [
        State("PhysicalActivities", "value"), State("HadAngina", "value"), State("HadStroke", "value"),
        State("HadAsthma", "value"), State("HadSkinCancer", "value"), State("HadCOPD", "value"),
        State("HadDepressiveDisorder", "value"), State("HadKidneyDisease", "value"), State("HadArthritis", "value"),
        State("HadDiabetes", "value"), State("DeafOrHardOfHearing", "value"), State("BlindOrVisionDifficulty", "value"),
        State("DifficultyConcentrating", "value"), State("DifficultyWalking", "value"), State("SleepHours", "value"),
        State("DifficultyDressingBathing", "value"), State("DifficultyErrands", "value"), State("ChestScan", "value"),
        State("AlcoholDrinkers", "value"), State("HIVTesting", "value"), State("HighRiskLastYear", "value"),
        State("GeneralHealth", "value"), State("SmokerStatus", "value"),
        State("HeightInMeters", "value"), State("WeightInKilograms", "value"), State("BMI", "value"),
        State("name_heart", "value"), State("sex_heart", "value"),
    ]
)
def predict(n_clicks, *values):
    if not n_clicks:
        return ""

    # Check for missing input
    if None in values:
        return "⚠️ Please fill all fields before predicting."

    # Extract input values
    name_heart = values[-2]
    sex_heart = values[-1]
    features = values[:-2] + (sex_heart,)  # combine all features including 'sex_heart'

    # Convert to numpy array for prediction
    input_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = heart_disease_model.predict(input_array)

    # Return formatted result
    if prediction[0] == 1:
        return html.Div(f"🩺 {name_heart} **HAS** heart disease!", className="alert alert-danger")
    else:
        return html.Div(f"✅ {name_heart} **does NOT have** heart disease.", className="alert alert-success")
    return ""


@dash.callback(
    Output("download-pdf-heart", "data"),
    Input("download-btn", "n_clicks"),
    [
        State("PhysicalActivities", "value"), State("HadAngina", "value"), State("HadStroke", "value"),
        State("HadAsthma", "value"), State("HadSkinCancer", "value"), State("HadCOPD", "value"),
        State("HadDepressiveDisorder", "value"), State("HadKidneyDisease", "value"), State("HadArthritis", "value"),
        State("HadDiabetes", "value"), State("DeafOrHardOfHearing", "value"), State("BlindOrVisionDifficulty", "value"),
        State("DifficultyConcentrating", "value"), State("DifficultyWalking", "value"), State("SleepHours", "value"),
        State("DifficultyDressingBathing", "value"), State("DifficultyErrands", "value"), State("ChestScan", "value"),
        State("AlcoholDrinkers", "value"), State("HIVTesting", "value"), State("HighRiskLastYear", "value"),
        State("GeneralHealth", "value"), State("SmokerStatus", "value"),
        State("HeightInMeters", "value"), State("WeightInKilograms", "value"), State("BMI", "value"),
        State("name_heart", "value"), State("sex_heart", "value"), State("prediction-result-heart", "children")
    ],
    prevent_initial_call=True
)
def download_report(n_clicks, *values):

    GeneralHealth, PhysicalActivities, SleepHours, HadAngina, HadStroke, HadAsthma, HadSkinCancer, HadCOPD, HadDepressiveDisorder, HadKidneyDisease, HadArthritis, HadDiabetes, DeafOrHardOfHearing, BlindOrVisionDifficulty, DifficultyConcentrating, DifficultyWalking, DifficultyDressingBathing, DifficultyErrands, ChestScan, AlcoholDrinkers, SmokerStatus, HIVTesting, HighRiskLastYear, HeightInMeters, WeightInKilograms, BMI, name_heart, sex_heart, prediction_result  = values

    if not n_clicks or None in [GeneralHealth, PhysicalActivities, SleepHours, HadAngina, HadStroke, HadAsthma, HadSkinCancer, HadCOPD, HadDepressiveDisorder, HadKidneyDisease, HadArthritis, HadDiabetes, DeafOrHardOfHearing, BlindOrVisionDifficulty, DifficultyConcentrating, DifficultyWalking, DifficultyDressingBathing, DifficultyErrands, ChestScan, AlcoholDrinkers, SmokerStatus, HIVTesting, HighRiskLastYear, HeightInMeters, WeightInKilograms, BMI, name_heart, sex_heart, prediction_result]:
        return None

    if isinstance(prediction_result, dict) and "props" in prediction_result:
        prediction_text = prediction_result["props"].get("children", "Prediction not available")
    else:
        prediction_text = str(prediction_result)

    labels_values = [
        ("Name", name_heart), 
        ("Sex", "Male" if sex_heart == 1 else "Female"),
        ("Physical Activities", "No" if PhysicalActivities == 0 else "Yes"), 
        ("Had Angina", "No" if HadAngina == 0 else "Yes"), 
        ("Had Stroke", "No" if HadStroke == 0 else "Yes"), 
        ("Had Asthma", "No" if HadAsthma == 0 else "Yes"), 
        ("Had Skin Cancer", "No" if HadSkinCancer == 0 else "Yes"), 
        ("Had COPD", "No" if HadCOPD == 0 else "Yes"),
        ("Had Depressive Disorder", "No" if HadDepressiveDisorder == 0 else "Yes"), 
        ("Had Kidney Disease", "No" if HadKidneyDisease == 0 else "Yes"), 
        ("Had Arthritis", "No" if HadArthritis == 0 else "Yes"), 
        ("Had Diabetes", "No" if HadDiabetes == 0 else "Yes"),
        ("Deaf or Hard of Hearing", "No" if DeafOrHardOfHearing == 0 else "Yes"), 
        ("Blind or Vision Difficulty", "No" if BlindOrVisionDifficulty == 0 else "Yes"), 
        ("Difficulty Concentrating", "No" if DifficultyConcentrating == 0 else "Yes"),
        ("Difficulty Walking", "No" if DifficultyWalking == 0 else "Yes"), 
        ("Difficulty Dressing/Bathing", "No" if DifficultyDressingBathing == 0 else "Yes"), 
        ("Sleep Hours", SleepHours), 
        ("Difficulty with Errands", "No" if DifficultyErrands == 0 else "Yes"), 
        ("Chest Scan", "No" if ChestScan == 0 else "Yes"),
        ("Alcohol Drinkers", "No" if AlcoholDrinkers == 0 else "Yes"), 
        ("HIV Testing", "No" if HIVTesting == 0 else "Yes"), 
        ("High Risk Last Year", "No" if HighRiskLastYear == 0 else "Yes"), 
        ("General Health", {0: "Excellent",
                            1: "Fair",
                            2: "Good",
                            3: "Poor",
                            4: "Very Good"}.get(GeneralHealth, "Unknown")), 
        ("Smoker Status", {0: "Every Day",
                            3: "Never Smoked",
                            1: "Some Days",
                            2: "Former"}.get(SmokerStatus, "Unknown")),
        ("Height (m)", WeightInKilograms), 
        ("Weight (kg)", HeightInMeters), 
        ("BMI", BMI),
    ]

    # Generate PDF
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)

    y = 750
    pdf.drawString(100, y, "Heart Disease Prediction Report")
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

    return dcc.send_bytes(buffer.getvalue(), filename=f"{name_heart}_Heart_Disease_Report.pdf")

@dash.callback(
    Output("PhysicalActivities", "value"), Output("HadAngina", "value"), Output("HadStroke", "value"),
    Output("HadAsthma", "value"), Output("HadSkinCancer", "value"), Output("HadCOPD", "value"),
    Output("HadDepressiveDisorder", "value"), Output("HadKidneyDisease", "value"), Output("HadArthritis", "value"),
    Output("HadDiabetes", "value"), Output("DeafOrHardOfHearing", "value"), Output("BlindOrVisionDifficulty", "value"),
    Output("DifficultyConcentrating", "value"), Output("DifficultyWalking", "value"), Output("SleepHours", "value"),
    Output("DifficultyDressingBathing", "value"), Output("DifficultyErrands", "value"), Output("ChestScan", "value"),
    Output("AlcoholDrinkers", "value"), Output("HIVTesting", "value"), Output("HighRiskLastYear", "value"),
    Output("GeneralHealth", "value"), Output("SmokerStatus", "value"),
    Output("HeightInMeters", "value"), Output("WeightInKilograms", "value"), Output("BMI", "value"),
    Output("name_heart", "value"), Output("sex_heart", "value"),
    Input("fill-random-btn", "n_clicks"),
    prevent_initial_call=True
)

def fill_from_dataset_heart(n_clicks):
    # Randomly sample one row from your dataframe (make sure your dataframe is name_heartd 'df')
    sample = heart_df.sample(1).iloc[0]

    # For 'name_heart', random pick from a list (or use sample if you have name_heart column)
    name_heart = random.choice(["Alex", "Sam", "Jamie", "Taylor", "Jordan", "Riya", "Karan", "Aisha"])

    # Map sex_heart if stored as string in dataset
    sex_heart_map = {'Male': 1, 'Female': 0}
    sex_heart = sex_heart_map.get(str(sample['Sex']), 0)
    Binary_map = { "Yes": 1, "No": 0}
    PhysicalActivities = Binary_map.get(str(sample["PhysicalActivities"]), 0) 
    HadAngina = Binary_map.get(str(sample["HadAngina"]), 0)
    HadStroke = Binary_map.get(str(sample["HadStroke"]), 0)
    HadAsthma = Binary_map.get(str(sample["HadAsthma"]), 0)
    HadSkinCancer = Binary_map.get(str(sample["HadSkinCancer"]), 0)
    HadCOPD = Binary_map.get(str(sample["HadCOPD"]), 0)
    HadDepressiveDisorder = Binary_map.get(str(sample["HadDepressiveDisorder"]), 0)
    HadKidneyDisease = Binary_map.get(str(sample["HadKidneyDisease"]), 0) 
    HadArthritis = Binary_map.get(str(sample["HadArthritis"]), 0)
    HadDiabetes = Binary_map.get(str(sample["HadDiabetes"]), 0)
    DeafOrHardOfHearing = Binary_map.get(str(sample["DeafOrHardOfHearing"]), 0)
    BlindOrVisionDifficulty = Binary_map.get(str(sample["BlindOrVisionDifficulty"]), 0)
    DifficultyConcentrating = Binary_map.get(str(sample["DifficultyConcentrating"]), 0)
    DifficultyWalking = Binary_map.get(str(sample["DifficultyWalking"]), 0)
    SleepHours = int(sample["SleepHours"])
    DifficultyDressingBathing = Binary_map.get(str(sample["DifficultyDressingBathing"]), 0)
    DifficultyErrands = Binary_map.get(str(sample["DifficultyErrands"]), 0)
    ChestScan = Binary_map.get(str(sample["ChestScan"]), 0)
    AlcoholDrinkers = Binary_map.get(str(sample["AlcoholDrinkers"]), 0)
    HIVTesting = Binary_map.get(str(sample["HIVTesting"]), 0)
    HighRiskLastYear = Binary_map.get(str(sample["HighRiskLastYear"]), 0)
    GeneralHealth_map = {0: "Excellent",
                            1: "Fair",
                            2: "Good",
                            3: "Poor",
                            4: "Very good"}
    GeneralHealth = GeneralHealth_map.get(str(sample["GeneralHealth"]), 0)
    SmokerStatus_map = {0: "Current smoker - now smokes every day",
                            3: "Never smoked",
                            1: "Current smoker - now smokes some days",
                            2: "Former smoker"}
    SmokerStatus = SmokerStatus_map.get(str(sample["SmokerStatus"]), 0)
    HeightInMeters = float(sample["HeightInMeters"])
    WeightInKilograms = float(sample["WeightInKilograms"])
    BMI = float(sample["BMI"])

    return PhysicalActivities, HadAngina, HadStroke, HadAsthma, HadSkinCancer, HadCOPD, HadDepressiveDisorder, HadKidneyDisease, HadArthritis, HadDiabetes, DeafOrHardOfHearing, BlindOrVisionDifficulty, DifficultyConcentrating, DifficultyWalking, SleepHours, DifficultyDressingBathing, DifficultyErrands, ChestScan, AlcoholDrinkers, HIVTesting, HighRiskLastYear, GeneralHealth, SmokerStatus, HeightInMeters, WeightInKilograms, BMI, name_heart, sex_heart