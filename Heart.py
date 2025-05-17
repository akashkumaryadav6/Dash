import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import joblib
import numpy as np
import base64
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load the trained model
heart_disease_model = joblib.load("./models/heart_disease_model.sav")


heart = dbc.Container(
    dbc.Card(
        dbc.CardBody([
            html.H2("Heart Disease Prediction", className="text-center mt-4 mb-4", style={"fontWeight": "bold"}),

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
                dbc.Col(dcc.Input(id="name", type="text", placeholder="Name", className="form-control"), md=6),
                dbc.Col(dcc.Dropdown(id="sex", options = [{'label':'Female', 'value':0}, {'label':'Male', 'value':1}], placeholder='Sex'), md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dbc.Button("Get Prediction", id="predict-btn", color="primary", className="mt-3"), md=6),
                dbc.Col(dbc.Button("Download Report", id="download-btn", color="success", className="mt-3"), md=6),
            ], className="mb-3"),

            html.Div(id="prediction-result", className="mt-4 text-center"),
            dcc.Download(id="download-pdf")
        ]),
        className="shadow p-4",
        style={"borderRadius": "15px", "maxWidth": "1000px", "marginTop": "5%"}
    ),
    fluid=True,
    className="d-flex flex-column justify-content-center align-items-center"
)

@dash.callback(
    Output("prediction-result", "children"),
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
        State("name", "value"), State("sex", "value"),
    ]
)
def predict(n_clicks, *values):
    if not n_clicks:
        return ""

    # Check for missing input
    if None in values:
        return "⚠️ Please fill all fields before predicting."

    # Extract input values
    name = values[-2]
    sex = values[-1]
    features = values[:-2] + (sex,)  # combine all features including 'sex'

    # Convert to numpy array for prediction
    input_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = heart_disease_model.predict(input_array)

    # Return formatted result
    if prediction[0] == 1:
        return html.Div(f"🩺 {name} **HAS** heart disease!", className="alert alert-danger")
    else:
        return html.Div(f"✅ {name} **does NOT have** heart disease.", className="alert alert-success")
    return ""


@dash.callback(
    Output("download-pdf", "data"),
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
        State("name", "value"), State("sex", "value"), State("prediction-result", "children")
    ],
    prevent_initial_call=True
)
def download_report(n_clicks, PhysicalActivities, HadAngina, HadStroke, HadAsthma, HadSkinCancer, HadCOPD,
                    HadDepressiveDisorder, HadKidneyDisease, HadArthritis, HadDiabetes,DeafOrHardOfHearing, 
                    BlindorVisionDifficulty, DifficultyConcentrating, DifficultyWalking, DifficultyDressingBathing, 
                    SleepHours, DifficultyErrands, ChestScan, AlcoholDrinkers, HIVTesting, HighRiskLastYear, 
                    GeneralHealth, SmokerStatus, HeightInMeters, WeightInKilograms, BMI, name, sex):
    values = [PhysicalActivities, HadAngina, HadStroke, HadAsthma, HadSkinCancer, HadCOPD,
                    HadDepressiveDisorder, HadKidneyDisease, HadArthritis, HadDiabetes,DeafOrHardOfHearing, 
                    BlindorVisionDifficulty, DifficultyConcentrating, DifficultyWalking, DifficultyDressingBathing, 
                    SleepHours, DifficultyErrands, ChestScan, AlcoholDrinkers, HIVTesting, HighRiskLastYear, 
                    GeneralHealth, SmokerStatus, HeightInMeters, WeightInKilograms, BMI, name, sex]
    if not n_clicks or None in values:
        return None

    *fields, name, prediction_result = values

    if isinstance(prediction_result, dict) and "props" in prediction_result:
        prediction_text = prediction_result["props"].get("children", "Prediction not available")
    else:
        prediction_text = str(prediction_result)

    labels = [
        "Physical Activities", "Had Angina", "Had Stroke", "Had Asthma", "Had Skin Cancer", "Had COPD",
        "Had Depressive Disorder", "Had Kidney Disease", "Had Arthritis", "Had Diabetes",
        "Deaf or Hard of Hearing", "Blind or Vision Difficulty", "Difficulty Concentrating",
        "Difficulty Walking", "Difficulty Dressing/Bathing", "Sleep Hours", "Difficulty with Errands", "Chest Scan",
        "Alcohol Drinkers", "HIV Testing", "High Risk Last Year", "General Health", "Smoker Status",
        "Height (m)", "Weight (kg)", "BMI", "Name", "Sex"
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

    for label, value in zip(labels, fields):
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

    return dcc.send_bytes(buffer.getvalue(), filename="Heart_Disease_Report.pdf")
