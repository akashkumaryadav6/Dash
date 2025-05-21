import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import joblib
import numpy as np
import pandas as pd
import base64
import io
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

kidney_df = pd.read_csv("datasets/kidney_disease_dataset.csv")
kidney_df.dropna(inplace=True)
kidney_model = joblib.load("./models/kidney_disease_model.sav")

kidney = dbc.Container(
    dbc.Card(
        dbc.CardBody([
            html.H2("Kidney Disease Prediction", className="text-center mt-4 mb-4", style={"fontWeight": "bold"}),

            dbc.Row([
                dbc.Col(dbc.Button("Fill Random Data", id="fill-random-btn", color="secondary", className="mb-4"), width="auto")
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Input(id="name", type="text", placeholder="Name", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="age_kidney", type="number", placeholder="Age of the patient", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="bp", type="number", placeholder="Blood pressure (mm/Hg)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="sg", type="number", step=0.01, placeholder="Specific gravity of urine", className="form-control"), md=3),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Input(id="al", type="number", placeholder="Albumin in urine", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="su", type="number", placeholder="Sugar in urine", className="form-control"), md=3),
                dbc.Col(dcc.Dropdown(id="rbc", options = [{'label':'Normal', 'value':0}, {'label':'Abnormal', 'value':1}], placeholder="Red blood cells in urine"), md=3),
                dbc.Col(dcc.Dropdown(id="pc", options = [{'label':'Normal', 'value':0}, {'label':'Abnormal', 'value':1}], placeholder="Pus cells in urine"), md=3),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="pcc", options = [{'label':'No', 'value':0}, {'label':'Yes', 'value':1}], placeholder="Pus cell clumps in urine"), md=3),
                dbc.Col(dcc.Dropdown(id="ba", options = [{'label':'No', 'value':0}, {'label':'Yes', 'value':1}], placeholder="Bacteria in urine"), md=3),
                dbc.Col(dcc.Input(id="bgr", type="number", placeholder="Random blood glucose level (mg/dl)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="bu", type="number", placeholder="Blood urea (mg/dl)", className="form-control"), md=3),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Input(id="sc", type="number", placeholder="Serum creatinine (mg/dl)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="sodium", type="number", placeholder="Sodium level (mEq/L)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="potassium", type="number", placeholder="Potassium level (mEq/L)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="hemo", type="number", placeholder="Hemoglobin level (gms)", className="form-control"), md=3),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Input(id="pcv", type="number", placeholder="Packed cell volume (%)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="wbcc", type="number", placeholder="White blood cell count (cells/cumm)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="rbcc", type="number", placeholder="Red blood cell count (millions/cumm)", className="form-control"), md=3),
                dbc.Col(dcc.Dropdown(id="htn", options = [{'label':'No', 'value':0}, {'label':'Yes', 'value':1}], placeholder="Hypertension"), md=3),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="dm", options = [{'label':'No', 'value':0}, {'label':'Yes', 'value':1}], placeholder="Diabetes mellitus"), md=3),
                dbc.Col(dcc.Dropdown(id="cad", options = [{'label':'No', 'value':0}, {'label':'Yes', 'value':1}], placeholder="Coronary artery disease"), md=3),
                dbc.Col(dcc.Dropdown(id="appet", options = [{'label':'Poor', 'value':0}, {'label':'Good', 'value':1}], placeholder="Appetite (Good/Poor)"), md=3),
                dbc.Col(dcc.Dropdown(id="ane", options = [{'label':'No', 'value':0}, {'label':'Yes', 'value':1}], placeholder="Anemia"), md=3),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Input(id="egfr", type="number", placeholder="Estimated GFR", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="urine_protein_creatinine_ratio", type="number", placeholder="Urine Protein/Creatinine Ratio", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="urine_output", type="number", placeholder="Urine output (ml/day)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="serum_albumin", type="number", placeholder="Serum albumin (g/dL)", className="form-control"), md=3),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Input(id="cholesterol", type="number", placeholder="Cholesterol level (mg/dL)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="pth", type="number", placeholder="Parathyroid hormone level (pg/mL)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="calcium", type="number", placeholder="Calcium level (mg/dL)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="phosphate", type="number", placeholder="Phosphate level (mg/dL)", className="form-control"), md=3),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Dropdown(id="family_history", options = [{'label':'No', 'value':0}, {'label':'Yes', 'value':1}], placeholder="Family history of kidney disease"), md=3),
                dbc.Col(dcc.Dropdown(id="smoking_status", options = [{'label':'No', 'value':0}, {'label':'Yes', 'value':1}], placeholder="Smoking status"), md=3),
                dbc.Col(dcc.Input(id="bmi_kidney", type="number", placeholder="Body Mass Index (kg/m²)", className="form-control"), md=3),
                dbc.Col(dcc.Dropdown(id="physical_activity", options = [{'label':'Low', 'value':0}, {'label':'Moderate', 'value':1}, {'label':'High', 'value': 2}], placeholder="Physical activity"), md=3),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Input(id="dm_duration", type="number", placeholder="Duration of diabetes (years)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="htn_duration", type="number", placeholder="Duration of hypertension (years)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="cystatin_c", type="number", placeholder="Cystatin C level (mg/L)", className="form-control"), md=3),
                dbc.Col(dcc.Input(id="urinary_sediment", type="number", placeholder="Urinary sediment (cells/hpf)", className="form-control"), md=3),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dcc.Input(id="crp", type="number", placeholder="C-Reactive Protein (mg/L)", className="form-control"), md=6),
                dbc.Col(dcc.Input(id="il6", type="number", placeholder="Interleukin-6 (pg/mL)", className="form-control"), md=6),
            ], className="mb-1"),

            dbc.Row([
                dbc.Col(dbc.Button("Get Prediction", id="predict-btn", color="primary", className="mt-3"), md=6),
                dbc.Col(dbc.Button("Download Report", id="download-btn", color="success", className="mt-3"), md=6),
            ], className="mb-1"),

            html.Div(id="prediction-result-kidney", className="mt-4 text-center"),
            dcc.Download(id="download-pdf-kidney")
        ], className="red-gradient"),
        className="shadow p-4 red-gradient",
        style={"borderRadius": "15px", "maxWidth": "1000px", "marginTop": "5%"}
    ),
    fluid=True,
    className="d-flex flex-column justify-content-center align-items-center"
)

@dash.callback(
    Output("prediction-result-kidney", "children"),
    Input("predict-btn", "n_clicks"),
    [
    State("name", "value"), State("age_kidney", "value"), State("bp", "value"), State("sg", "value"), 
    State("al", "value"), State("su", "value"), State("rbc", "value"), State("pc", "value"), 
    State("pcc", "value"), State("ba", "value"), State("bgr", "value"), State("bu", "value"),
    State("sc", "value"), State("sodium", "value"),State("potassium", "value"), State("hemo", "value"),
    State("pcv", "value"), State("wbcc", "value"), State("rbcc", "value"), State("htn", "value"),
    State("dm", "value"), State("cad", "value"), State("appet", "value"), State("ane", "value"),
    State("egfr", "value"), State("urine_protein_creatinine_ratio", "value"), State("urine_output", "value"), 
    State("serum_albumin", "value"), State("cholesterol", "value"), State("pth", "value"), State("calcium", "value"),        
    State("phosphate", "value"), State("family_history", "value"), State("smoking_status", "value"), State("bmi_kidney", "value"),        
    State("physical_activity", "value"), State("dm_duration", "value"), State("htn_duration", "value"), 
    State("cystatin_c", "value"), State("urinary_sediment", "value"), State("crp", "value"), 
    State("il6", "value")],
    prevent_initial_call=True
)

def predict(n_clicks, *values):
    name, age_kidney, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sodium, potassium, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, ane, egfr, urine_protein_creatinine_ratio, urine_output, serum_albumin, cholesterol, pth, calcium, phosphate, family_history, smoking_status,bmi_kidney, physical_activity, dm_duration, htn_duration, cystatin_c, urinary_sediment, crp, il6 = values
    if not n_clicks:
        return ""
    
    # Check for missing inputs
    if None in [name, age_kidney, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sodium, potassium, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, ane, egfr, urine_protein_creatinine_ratio, urine_output, serum_albumin, cholesterol, pth, calcium, phosphate, family_history, smoking_status,bmi_kidney, physical_activity, dm_duration, htn_duration, cystatin_c, urinary_sediment, crp, il6]:
        return html.Div("⚠️ Please fill all fields before predicting.", className="alert alert-warning")

    name = values[0]
    # Create input array
    features = np.array([[age_kidney, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sodium, potassium, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, ane, egfr, urine_protein_creatinine_ratio, urine_output, serum_albumin, cholesterol, pth, calcium, phosphate, family_history, smoking_status,bmi_kidney, physical_activity, dm_duration, htn_duration, cystatin_c, urinary_sediment, crp, il6]])

    # Predict using preloaded model
    prediction = kidney_model.predict(features)
    # Return result
    if prediction[0] == 0:
        return html.Div(f"✅ {name}, you are **not likely** to have Kidney Disease.", className="alert alert-success")
    if prediction[0] == 1:
        return html.Div(f"🩺 {name}, you have **LOW RISK** of Kidney Disease.", className="alert alert-danger")
    if prediction[0] == 2:
        return html.Div(f"🩺 {name}, you have **MODERATE RISK** of Kidney Disease.", className="alert alert-danger")
    if prediction[0] == 3:
        return html.Div(f"🩺 {name}, you have **SEVERE RISK** Kidney Disease.", className="alert alert-danger")
    if prediction[0] == 4:
        return html.Div(f"🩺 {name}, you have **HIGH RISK** Kidney Disease.", className="alert alert-danger")
    return ''

@dash.callback(
    Output("download-pdf-kidney", "data"),
    Input("download-btn", "n_clicks"),
    [
    State("name", "value"), State("age_kidney", "value"), State("bp", "value"), State("sg", "value"), 
    State("al", "value"), State("su", "value"), State("rbc", "value"), State("pc", "value"), 
    State("pcc", "value"), State("ba", "value"), State("bgr", "value"), State("bu", "value"),
    State("sc", "value"), State("sodium", "value"),State("potassium", "value"), State("hemo", "value"),
    State("pcv", "value"), State("wbcc", "value"), State("rbcc", "value"), State("htn", "value"),
    State("dm", "value"), State("cad", "value"), State("appet", "value"), State("ane", "value"),
    State("egfr", "value"), State("urine_protein_creatinine_ratio", "value"), State("urine_output", "value"), 
    State("serum_albumin", "value"), State("cholesterol", "value"), State("pth", "value"), State("calcium", "value"),        
    State("phosphate", "value"), State("family_history", "value"), State("smoking_status", "value"), State("bmi_kidney", "value"),        
    State("physical_activity", "value"), State("dm_duration", "value"), State("htn_duration", "value"), 
    State("cystatin_c", "value"), State("urinary_sediment", "value"), State("crp", "value"), 
    State("il6", "value"), State("prediction-result-kidney", "children")
    ],
    prevent_initial_call=True
)

def download_kidney_report(n_clicks, *values):
    name, age_kidney, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sodium, potassium, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, ane, egfr, urine_protein_creatinine_ratio, urine_output, serum_albumin, cholesterol, pth, calcium, phosphate, family_history, smoking_status,bmi_kidney, physical_activity, dm_duration, htn_duration, cystatin_c, urinary_sediment, crp, il6, prediction_result_kidney = values
    if not n_clicks or None in [name, age_kidney, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sodium, potassium, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, ane, egfr, urine_protein_creatinine_ratio, urine_output, serum_albumin, cholesterol, pth, calcium, phosphate, family_history, smoking_status,bmi_kidney, physical_activity, dm_duration, htn_duration, cystatin_c, urinary_sediment, crp, il6, prediction_result_kidney]:
        return None

    # Handle Dash component in prediction_result_kidney
    if isinstance(prediction_result_kidney, dict) and "props" in prediction_result_kidney:
        prediction_text = prediction_result_kidney["props"].get("children", "Prediction not available")
    else:
        prediction_text = str(prediction_result_kidney)

    labels_values =  [
    ("Name", name),
    ("Age", age_kidney),
    ("Blood Pressure (mm/Hg)", bp),
    ("Specific gravity of urine", sg),
    ("Albumin in urine", al),
    ("Sugar in urine", su),
    ("Red blood cells in urine", "Abnormal" if rbc == 1 else "Normal"),
    ("Pus cells in urine", "Abnormal" if pc == 1 else "Normal"),
    ("Pus cell clumps in urine", "Present" if pcc == 1 else "Not Present"),
    ("Bacteria in urine", "Present" if ba == 1 else "Not Present"),
    ("Random blood glucose level (mg/dl)", bgr),
    ("Blood urea (mg/dl)", bu),
    ("Serum creatinine (mg/dl)", sc),
    ("Sodium level (mEq/L)", sodium),
    ("Potassium level (mEq/L)", potassium),
    ("Hemoglobin level (gms)", hemo),
    ("Packed cell volume (%)", pcv),
    ("White blood cell count (cells/cumm)", wbcc),
    ("Red blood cell count (millions/cumm)", rbcc),
    ("Hypertension", "Yes" if htn == 1 else "No"),
    ("Diabetes mellitus", "Yes" if dm == 1 else "No"),
    ("Coronary artery disease", "Yes" if cad == 1 else "No"),
    ("Appetite (Good/Poor)", "Good" if appet == 1 else "Poor"),
    ("Anemia", "Yes" if ane == 1 else "No"),
    ("Estimated Glomerular Filtration Rate (eGFR)", egfr),
    ("Urine protein-to-creatinine ratio", urine_protein_creatinine_ratio),
    ("Urine output (ml/day)", urine_output),
    ("Serum albumin level", serum_albumin),
    ("Cholesterol level", cholesterol),
    ("Parathyroid hormone (PTH) level", pth),
    ("Serum calcium level", calcium),
    ("Serum phosphate level", phosphate),
    ("Family history of chronic kidney disease", "Yes" if family_history == 1 else "No"),
    ("Smoking status", "Yes" if smoking_status == 1 else "No"),
    ("Body Mass Index (BMI)",bmi_kidney),
    ("Physical activity level", {0: "Low",
                                 1: "Moderate",
                                 2: "High"}.get(physical_activity, "Unknown")),
    ("Duration of diabetes mellitus (years)", dm_duration),
    ("Duration of hypertension (years)", htn_duration),
    ("Cystatin C level", cystatin_c),
    ("Urinary sediment microscopy results", "Abnormal" if urinary_sediment == 1 else "Normal"),
    ("C-reactive protein (CRP) level", crp),
    ("Interleukin-6 (IL-6) level", il6),
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

    return dcc.send_bytes(buffer.getvalue(), filename=f"{name}_Kidney_Disease_Prediction_Report.pdf")

@dash.callback(
        Output("name", "value"), Output("age_kidney", "value"), Output("bp", "value"), Output("sg", "value"), 
    Output("al", "value"), Output("su", "value"), Output("rbc", "value"), Output("pc", "value"), 
    Output("pcc", "value"), Output("ba", "value"), Output("bgr", "value"), Output("bu", "value"),
    Output("sc", "value"), Output("sodium", "value"),Output("potassium", "value"), Output("hemo", "value"),
    Output("pcv", "value"), Output("wbcc", "value"), Output("rbcc", "value"), Output("htn", "value"),
    Output("dm", "value"), Output("cad", "value"), Output("appet", "value"), Output("ane", "value"),
    Output("egfr", "value"), Output("urine_protein_creatinine_ratio", "value"), Output("urine_output", "value"), 
    Output("serum_albumin", "value"), Output("cholesterol", "value"), Output("pth", "value"), Output("calcium", "value"),        
    Output("phosphate", "value"), Output("family_history", "value"), Output("smoking_status", "value"), Output("bmi_kidney", "value"),        
    Output("physical_activity", "value"), Output("dm_duration", "value"), Output("htn_duration", "value"), 
    Output("cystatin_c", "value"), Output("urinary_sediment", "value"), Output("crp", "value"), 
    Output("il6", "value"),Input("fill-random-btn", "n_clicks"),
    prevent_initial_call=True
)

def fill_from_dataset_ckd(n_clicks):
    # Randomly sample one row
    sample = kidney_df.sample(1).iloc[0]

    # Binary map for yes/no type columns
    binary_map = {"yes": 1, "no": 0}
    appetite_map = {"good": 1, "poor": 0}
    activity_map = {"high": 2, "moderate": 1, "low": 0}
    map1 = {'abnormal': 1, 'normal': 0}
    map2 = {'present': 1, 'not present': 0}

    age_kidney = int(sample['Age of the patient'])
    bp = int(sample['Blood pressure (mm/Hg)'])
    sg = float(sample['Specific gravity of urine'])
    al = int(sample['Albumin in urine'])
    su = int(sample['Sugar in urine'])
    rbc = map1.get(str(sample['Red blood cells in urine']), 0)
    pc = map1.get(str(sample['Pus cells in urine']), 0)
    pcc = map2.get(str(sample['Pus cell clumps in urine']), 0)
    ba = map2.get(str(sample['Bacteria in urine']), 0)
    bgr = float(sample['Random blood glucose level (mg/dl)'])
    bu = float(sample['Blood urea (mg/dl)'])
    sc = float(sample['Serum creatinine (mg/dl)'])
    sodium = float(sample['Sodium level (mEq/L)'])
    potassium = float(sample['Potassium level (mEq/L)'])
    hemo = float(sample['Hemoglobin level (gms)'])
    pcv = float(sample['Packed cell volume (%)'])
    wbcc = float(sample['White blood cell count (cells/cumm)'])
    rbcc = float(sample['Red blood cell count (millions/cumm)'])
    htn = binary_map.get(str(sample['Hypertension (yes/no)']), 0)
    dm = binary_map.get(str(sample['Diabetes mellitus (yes/no)']), 0)
    cad = binary_map.get(str(sample['Coronary artery disease (yes/no)']), 0)
    appet = appetite_map.get(str(sample['Appetite (good/poor)']), 1)
    ane = binary_map.get(str(sample['Anemia (yes/no)']), 0)
    egfr = float(sample['Estimated Glomerular Filtration Rate (eGFR)'])
    urine_protein_creatinine_ratio = float(sample['Urine protein-to-creatinine ratio'])
    urine_output = float(sample['Urine output (ml/day)'])
    serum_albumin = float(sample['Serum albumin level'])
    cholesterol = float(sample['Cholesterol level'])
    pth = float(sample['Parathyroid hormone (PTH) level'])
    calcium = float(sample['Serum calcium level'])
    phosphate = float(sample['Serum phosphate level'])
    family_history = binary_map.get(str(sample['Family history of chronic kidney disease']), 0)
    smoking_status = binary_map.get(str(sample['Smoking status']).lower(), 0)
    bmi_kidney = float(sample['Body Mass Index (BMI)'])
    physical_activity = activity_map.get(str(sample['Physical activity level']).lower(), 1)
    dm_duration = float(sample['Duration of diabetes mellitus (years)'])
    htn_duration = float(sample['Duration of hypertension (years)'])
    cystatin_c = float(sample['Cystatin C level'])
    urinary_sediment = map1.get(sample['Urinary sediment microscopy results'])
    crp = float(sample['C-reactive protein (CRP) level'])
    il6 = float(sample['Interleukin-6 (IL-6) level'])

    # You can randomize name if required
    name = random.choice(["Alex", "Sam", "Jamie", "Taylor", "Jordan", "Riya", "Karan", "Aisha"])

    return name, age_kidney, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sodium, potassium, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, ane, egfr, urine_protein_creatinine_ratio, urine_output, serum_albumin, cholesterol, pth, calcium, phosphate, family_history, smoking_status,bmi_kidney, physical_activity, dm_duration, htn_duration, cystatin_c, urinary_sediment, crp, il6