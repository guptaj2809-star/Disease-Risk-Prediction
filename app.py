import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Disease Prediction",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SYMPTOMS = [
    "fever",
    "high_fever",
    "cough",
    "fatigue",
    "headache",
    "body_pain",
    "joint_pain",
    "sore_throat",
    "runny_nose",
    "shortness_of_breath",
    "chest_pain",
    "wheezing",
    "loss_of_smell",
    "nausea",
    "vomiting",
    "diarrhea",
    "abdominal_pain",
    "bloating",
    "constipation",
    "increased_thirst",
    "frequent_urination",
    "blurred_vision",
    "itching",
    "skin_rash",
    "sneezing",
    "chills",
    "dizziness",
    "yellowing_of_skin",
    "burning_urination",
    "back_pain",
    "sweating",
    "ear_pain",
    "swollen_glands",
    "sinus_pressure",
    "night_sweats",
    "acid_reflux",
    "weight_loss",
    "pale_skin",
    "anxiety",
    "palpitations",
    "swelling_legs",
    "confusion",
    "neck_stiffness",
    "sensitivity_to_light",
    "hearing_loss",
    "dry_skin",
    "appetite_loss",
    "mood_changes",
    "cough_with_mucus",
    "chest_tightness",
    "rash_blisters",
    "abdominal_cramps",
    "dehydration",
    "muscle_weakness",
    "blood_in_urine",
    "sleep_disturbance",
    "sore_eyes",
    "fainting",
]

SYMPTOM_LABELS = {symptom: symptom.replace("_", " ").title() for symptom in SYMPTOMS}

DISEASE_INFO = {
    "Flu": "Often linked with fever, fatigue, body pain, chills, and cough.",
    "Common Cold": "Usually includes runny nose, sneezing, sore throat, and mild cough.",
    "COVID-19": "Can include fever, cough, fatigue, breathlessness, and loss of smell.",
    "Pneumonia": "Often shows cough, fever, chest pain, and shortness of breath.",
    "Asthma": "Common signs include wheezing, chest tightness, and shortness of breath.",
    "Bronchitis": "Often includes cough, fatigue, wheezing, and chest discomfort.",
    "Food Poisoning": "Typical pattern is nausea, vomiting, diarrhea, and abdominal pain.",
    "Gastritis": "Usually involves nausea, bloating, abdominal pain, and vomiting.",
    "Migraine": "Frequently includes headache, nausea, dizziness, and blurred vision.",
    "Diabetes": "Can involve increased thirst, frequent urination, blurred vision, and fatigue.",
    "Allergy": "Often presents with sneezing, runny nose, itching, and skin rash.",
    "Dengue": "Can show high fever, headache, joint pain, body pain, and rash.",
    "Malaria": "Frequently includes fever, chills, headache, sweating, and body pain.",
    "Typhoid": "Often linked with prolonged fever, abdominal pain, weakness, and headache.",
    "Urinary Tract Infection": "Common signs are burning urination, frequent urination, fever, and back pain.",
    "Hepatitis": "Can include fatigue, nausea, abdominal pain, and yellowing of skin.",
    "Sinusitis": "Often brings sinus pressure, headache, runny nose, and sore throat.",
    "Chickenpox": "Can present with fever, itching, body pain, and a skin rash.",
    "Measles": "Often involves fever, cough, runny nose, and widespread skin rash.",
    "Arthritis": "Commonly linked with joint pain, body pain, fatigue, and stiffness-like discomfort.",
    "Hypertension": "May be associated with headache, dizziness, chest discomfort, and fatigue.",
    "GERD": "Often includes acid reflux, chest discomfort, nausea, and bloating.",
    "Tuberculosis": "Can include cough, fever, weight loss, night sweats, and chest pain.",
    "Anemia": "Often presents with fatigue, dizziness, pale skin, shortness of breath, and headache.",
    "Heart Disease": "Often includes chest pain, shortness of breath, palpitations, fatigue, and sweating.",
    "Stroke": "Can involve confusion, dizziness, headache, blurred vision, and sudden weakness-like symptoms.",
    "Chronic Kidney Disease": "May include fatigue, swelling in legs, nausea, itching, and changes in urination.",
    "Appendicitis": "Often causes abdominal pain, nausea, vomiting, fever, and appetite loss.",
    "HIV/AIDS": "Can involve weight loss, fever, fatigue, night sweats, diarrhea, and swollen glands.",
    "Meningitis": "May present with high fever, headache, neck stiffness, vomiting, and sensitivity to light.",
    "Otitis Media": "Often includes ear pain, fever, hearing difficulty, sore throat, and headache.",
    "Psoriasis": "Can cause skin rash, itching, dry skin, and sometimes joint pain.",
    "Hypothyroidism": "Often linked with fatigue, dry skin, constipation, mood changes, and pale skin.",
    "Hyperthyroidism": "May include weight loss, palpitations, sweating, anxiety, and fatigue.",
    "COPD": "Common signs include cough, wheezing, shortness of breath, chest discomfort, and fatigue.",
}

CONDITION_THEME = {
    "Flu": ("Acute Viral Pattern", "#dbeafe", "#1d4ed8"),
    "Common Cold": ("Upper Airway Pattern", "#e0f2fe", "#0369a1"),
    "COVID-19": ("Respiratory Alert", "#fee2e2", "#b91c1c"),
    "Pneumonia": ("Pulmonary Concern", "#ede9fe", "#6d28d9"),
    "Asthma": ("Airway Reactivity", "#dcfce7", "#15803d"),
    "Bronchitis": ("Bronchial Inflammation", "#fef3c7", "#b45309"),
    "Food Poisoning": ("Digestive Distress", "#ffedd5", "#c2410c"),
    "Gastritis": ("Stomach Irritation", "#fae8ff", "#a21caf"),
    "Migraine": ("Neurological Pattern", "#ede9fe", "#7c3aed"),
    "Diabetes": ("Metabolic Pattern", "#cffafe", "#0f766e"),
    "Allergy": ("Immune Response", "#fce7f3", "#be185d"),
    "Heart Disease": ("Cardiac Risk", "#fee2e2", "#b91c1c"),
    "Stroke": ("Neurological Emergency Pattern", "#ede9fe", "#6d28d9"),
    "Chronic Kidney Disease": ("Renal Pattern", "#e0f2fe", "#0369a1"),
    "Appendicitis": ("Acute Abdominal Pattern", "#ffedd5", "#c2410c"),
    "HIV/AIDS": ("Systemic Infection Pattern", "#fdf2f8", "#be185d"),
    "Meningitis": ("CNS Alert Pattern", "#fef3c7", "#b45309"),
    "COPD": ("Chronic Respiratory Pattern", "#dcfce7", "#15803d"),
}


def make_record(disease: str, active_symptoms: list[str]) -> dict:
    record = {"disease": disease}
    for symptom in SYMPTOMS:
        record[symptom] = int(symptom in active_symptoms)
    return record


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    disease_profiles = {
        "Flu": {
            "core": ["fever", "cough", "fatigue", "body_pain", "chills"],
            "support": ["headache", "sore_throat", "runny_nose", "joint_pain"],
        },
        "Common Cold": {
            "core": ["cough", "sore_throat", "runny_nose", "sneezing"],
            "support": ["headache", "fatigue", "ear_pain", "sinus_pressure"],
        },
        "COVID-19": {
            "core": ["fever", "cough", "fatigue", "shortness_of_breath", "loss_of_smell"],
            "support": ["headache", "body_pain", "diarrhea", "chest_pain", "sore_eyes"],
        },
        "Pneumonia": {
            "core": ["fever", "cough", "shortness_of_breath", "chest_pain"],
            "support": ["fatigue", "high_fever", "chills", "body_pain"],
        },
        "Asthma": {
            "core": ["shortness_of_breath", "wheezing", "chest_pain"],
            "support": ["cough", "runny_nose", "fatigue", "anxiety", "chest_tightness"],
        },
        "Bronchitis": {
            "core": ["cough", "fatigue", "wheezing"],
            "support": ["chest_pain", "sore_throat", "shortness_of_breath", "runny_nose", "cough_with_mucus"],
        },
        "Food Poisoning": {
            "core": ["nausea", "vomiting", "diarrhea", "abdominal_pain"],
            "support": ["fatigue", "chills", "dizziness", "fever", "dehydration"],
        },
        "Gastritis": {
            "core": ["nausea", "abdominal_pain", "bloating"],
            "support": ["vomiting", "constipation", "acid_reflux", "fatigue", "abdominal_cramps"],
        },
        "Migraine": {
            "core": ["headache", "dizziness", "blurred_vision"],
            "support": ["nausea", "vomiting", "fatigue", "anxiety"],
        },
        "Diabetes": {
            "core": ["increased_thirst", "frequent_urination", "fatigue"],
            "support": ["blurred_vision", "itching", "weight_loss", "dizziness"],
        },
        "Allergy": {
            "core": ["runny_nose", "sneezing", "itching"],
            "support": ["skin_rash", "cough", "sinus_pressure", "swollen_glands"],
        },
        "Dengue": {
            "core": ["high_fever", "headache", "joint_pain", "body_pain"],
            "support": ["skin_rash", "chills", "nausea", "fatigue", "sore_eyes"],
        },
        "Malaria": {
            "core": ["high_fever", "chills", "sweating", "body_pain"],
            "support": ["headache", "fatigue", "dizziness", "nausea"],
        },
        "Typhoid": {
            "core": ["high_fever", "abdominal_pain", "fatigue"],
            "support": ["headache", "constipation", "diarrhea", "body_pain"],
        },
        "Urinary Tract Infection": {
            "core": ["burning_urination", "frequent_urination", "back_pain"],
            "support": ["fever", "abdominal_pain", "fatigue", "dizziness", "blood_in_urine"],
        },
        "Hepatitis": {
            "core": ["yellowing_of_skin", "abdominal_pain", "fatigue"],
            "support": ["nausea", "vomiting", "diarrhea", "fever"],
        },
        "Sinusitis": {
            "core": ["sinus_pressure", "headache", "runny_nose"],
            "support": ["sore_throat", "cough", "fatigue", "ear_pain"],
        },
        "Chickenpox": {
            "core": ["skin_rash", "itching", "fever"],
            "support": ["fatigue", "body_pain", "headache", "sore_throat", "rash_blisters"],
        },
        "Measles": {
            "core": ["fever", "cough", "runny_nose", "skin_rash"],
            "support": ["sore_throat", "fatigue", "swollen_glands", "headache"],
        },
        "Arthritis": {
            "core": ["joint_pain", "body_pain", "fatigue"],
            "support": ["back_pain", "swollen_glands", "headache", "dizziness"],
        },
        "Hypertension": {
            "core": ["headache", "dizziness"],
            "support": ["chest_pain", "fatigue", "blurred_vision", "anxiety"],
        },
        "GERD": {
            "core": ["acid_reflux", "chest_pain", "bloating"],
            "support": ["nausea", "abdominal_pain", "cough", "sore_throat"],
        },
        "Tuberculosis": {
            "core": ["cough", "weight_loss", "night_sweats", "fatigue"],
            "support": ["fever", "chest_pain", "shortness_of_breath", "high_fever"],
        },
        "Anemia": {
            "core": ["fatigue", "dizziness", "pale_skin"],
            "support": ["shortness_of_breath", "headache", "weight_loss", "anxiety"],
        },
        "Heart Disease": {
            "core": ["chest_pain", "shortness_of_breath", "palpitations"],
            "support": ["fatigue", "sweating", "dizziness", "swelling_legs", "chest_tightness"],
        },
        "Stroke": {
            "core": ["confusion", "dizziness", "headache"],
            "support": ["blurred_vision", "anxiety", "palpitations", "fatigue", "muscle_weakness"],
        },
        "Chronic Kidney Disease": {
            "core": ["fatigue", "swelling_legs", "nausea"],
            "support": ["frequent_urination", "back_pain", "itching", "pale_skin", "blood_in_urine"],
        },
        "Appendicitis": {
            "core": ["abdominal_pain", "nausea", "appetite_loss"],
            "support": ["vomiting", "fever", "constipation", "diarrhea", "abdominal_cramps"],
        },
        "HIV/AIDS": {
            "core": ["weight_loss", "fatigue", "night_sweats"],
            "support": ["fever", "diarrhea", "swollen_glands", "skin_rash", "appetite_loss"],
        },
        "Meningitis": {
            "core": ["high_fever", "headache", "neck_stiffness"],
            "support": ["vomiting", "sensitivity_to_light", "confusion", "fatigue", "fainting"],
        },
        "Otitis Media": {
            "core": ["ear_pain", "fever", "hearing_loss"],
            "support": ["headache", "sore_throat", "runny_nose", "swollen_glands"],
        },
        "Psoriasis": {
            "core": ["skin_rash", "itching", "dry_skin"],
            "support": ["joint_pain", "fatigue", "body_pain", "mood_changes"],
        },
        "Hypothyroidism": {
            "core": ["fatigue", "dry_skin", "constipation"],
            "support": ["pale_skin", "mood_changes", "dizziness", "sleep_disturbance", "muscle_weakness"],
        },
        "Hyperthyroidism": {
            "core": ["weight_loss", "palpitations", "sweating"],
            "support": ["anxiety", "fatigue", "diarrhea", "dizziness", "sleep_disturbance"],
        },
        "COPD": {
            "core": ["cough", "wheezing", "shortness_of_breath"],
            "support": ["chest_pain", "fatigue", "weight_loss", "anxiety", "cough_with_mucus", "chest_tightness"],
        },
    }

    records = []
    cases_per_disease = 70
    for disease, profile in disease_profiles.items():
        core = profile["core"]
        support = profile["support"]
        other_symptoms = [symptom for symptom in SYMPTOMS if symptom not in core + support]

        for _ in range(cases_per_disease):
            active = set()
            for symptom in core:
                if rng.random() < 0.92:
                    active.add(symptom)
            for symptom in support:
                if rng.random() < 0.45:
                    active.add(symptom)
            noise_count = int(rng.integers(0, 3))
            if noise_count:
                noise = rng.choice(other_symptoms, size=noise_count, replace=False)
                active.update(noise.tolist())
            if not active:
                active.add(core[0])
            records.append(make_record(disease, sorted(active)))

    return pd.DataFrame(records)


@st.cache_resource
def train_model():
    data = build_dataset()
    x = data[SYMPTOMS]
    y = data["disease"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_count = len(encoder.classes_)
    test_size = max(class_count * 3, int(len(data) * 0.25))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    model = Pipeline(
        steps=[
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=16,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                ),
            )
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    decoded_true = encoder.inverse_transform(y_test)
    decoded_pred = encoder.inverse_transform(predictions)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(decoded_true, decoded_pred, labels=encoder.classes_),
            index=encoder.classes_,
            columns=encoder.classes_,
        ),
        "report": classification_report(
            decoded_true, decoded_pred, output_dict=True, zero_division=0
        ),
    }
    return model, encoder, data, metrics


model, encoder, dataset, metrics = train_model()


def render_score_bar(label: str, value: float, color: str) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="font-weight:700; color:#ffffff;">{label}</span>
                <span style="color:#d4d4d8;">{value:.2f}%</span>
            </div>
            <div style="height:12px; background:#2a2a32; border-radius:999px; overflow:hidden;">
                <div style="width:{min(value, 100):.2f}%; height:12px; background:{color}; border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 14% 10%, rgba(239, 68, 68, 0.12), transparent 22%),
            radial-gradient(circle at 86% 16%, rgba(185, 28, 28, 0.12), transparent 20%),
            linear-gradient(180deg, #0b0b0d 0%, #17171b 100%);
        color: #f5f5f5;
    }
    [data-testid="stSidebar"] {display:none;}
    [data-testid="collapsedControl"] {display:none;}
    .hero {
        background:
            radial-gradient(circle at 22% 22%, rgba(255,255,255,0.10), transparent 20%),
            radial-gradient(circle at 80% 18%, rgba(255,255,255,0.08), transparent 18%),
            linear-gradient(135deg, #09090b 0%, #4a0d0d 52%, #b91c1c 100%);
        border-radius: 34px;
        padding: 38px;
        color: white;
        box-shadow: 0 28px 70px rgba(127, 29, 29, 0.28);
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 2.7rem;
        font-weight: 800;
    }
    .hero p {
        margin: 0;
        opacity: 0.95;
        font-size: 1.03rem;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1.7fr 1fr;
        gap: 18px;
        align-items: center;
    }
    .info-card {
        background: rgba(22, 22, 26, 0.92);
        border: 1px solid rgba(239, 68, 68, 0.12);
        border-radius: 28px;
        padding: 22px 24px;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.24);
        margin-bottom: 18px;
    }
    .result-card {
        background:
            radial-gradient(circle at top right, rgba(239, 68, 68, 0.18), transparent 28%),
            radial-gradient(circle at bottom left, rgba(120, 13, 13, 0.16), transparent 24%),
            linear-gradient(135deg, #111114 0%, #1b1b21 100%);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 30px;
        padding: 26px;
        box-shadow: 0 20px 40px rgba(127, 29, 29, 0.24);
        margin: 14px 0 18px 0;
    }
    .result-label {
        color: #fca5a5;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }
    .result-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .mini-stat {
        background: rgba(20,20,24,0.94);
        border-radius: 22px;
        padding: 16px 18px;
        border: 1px solid rgba(239, 68, 68, 0.12);
        text-align: center;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.24);
    }
    .mini-stat h3 {
        margin: 0;
        color: #ffffff;
        font-size: 1.6rem;
    }
    .mini-stat p {
        margin: 4px 0 0 0;
        color: #d4d4d8;
        font-size: 0.95rem;
    }
    .feature-pill {
        display: inline-block;
        padding: 9px 13px;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .disease-cloud {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .disease-chip {
        background: #221012;
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.14);
        border-radius: 999px;
        padding: 8px 13px;
        font-size: 0.9rem;
    }
    .selected-chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
    }
    .selected-chip {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        color: white;
        border-radius: 999px;
        padding: 9px 14px;
        font-size: 0.92rem;
        box-shadow: 0 10px 22px rgba(220, 38, 38, 0.24);
    }
    .glass-card {
        background: rgba(20,20,24,0.78);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(239, 68, 68, 0.12);
        border-radius: 28px;
        padding: 20px 22px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
        margin-bottom: 18px;
    }
    .spotlight {
        border-radius: 30px;
        padding: 26px;
        background: linear-gradient(135deg, #1a1010 0%, #2b1515 100%);
        border: 1px solid rgba(239, 68, 68, 0.16);
        box-shadow: 0 18px 38px rgba(127, 29, 29, 0.2);
    }
    .page-title {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .page-copy {
        color: #d4d4d8;
        margin-bottom: 18px;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 14px;
    }
    .feature-box {
        background: rgba(20,20,24,0.92);
        border: 1px solid rgba(239, 68, 68, 0.12);
        border-radius: 24px;
        padding: 20px;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.24);
    }
    .feature-box h4 {
        margin: 0 0 8px 0;
        color: #ffffff;
    }
    .feature-box p {
        margin: 0;
        color: #d4d4d8;
        font-size: 0.95rem;
    }
    .insight-banner {
        background: linear-gradient(135deg, #09090b 0%, #7f1d1d 58%, #dc2626 100%);
        border-radius: 32px;
        padding: 28px;
        color: white;
        box-shadow: 0 22px 48px rgba(127, 29, 29, 0.24);
        margin-bottom: 18px;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    .metric-panel {
        background: rgba(20,20,24,0.92);
        border-radius: 24px;
        padding: 20px;
        border: 1px solid rgba(239, 68, 68, 0.12);
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.24);
    }
    .metric-panel h4 {
        margin: 0 0 8px 0;
        color: #ffffff;
    }
    .metric-panel p {
        margin: 0;
        color: #d4d4d8;
    }
    .timeline {
        border-left: 3px solid rgba(239, 68, 68, 0.45);
        padding-left: 16px;
        margin-top: 8px;
    }
    .timeline-item {
        margin-bottom: 16px;
    }
    .timeline-item h5 {
        margin: 0 0 6px 0;
        color: #ffffff;
        font-size: 1rem;
    }
    .timeline-item p {
        margin: 0;
        color: #d4d4d8;
        font-size: 0.94rem;
    }
    .page-hero {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 18px;
        margin-bottom: 18px;
    }
    .page-panel {
        background: rgba(20,20,24,0.9);
        border: 1px solid rgba(239, 68, 68, 0.12);
        border-radius: 28px;
        padding: 22px 24px;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.24);
    }
    .prediction-grid {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 18px;
        align-items: start;
    }
    .score-card {
        background: rgba(20,20,24,0.92);
        border: 1px solid rgba(239, 68, 68, 0.12);
        border-radius: 22px;
        padding: 18px 20px;
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.24);
        margin-bottom: 14px;
    }
    .score-big {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 4px 0;
    }
    .section-note {
        color: #d4d4d8;
        font-size: 0.96rem;
        line-height: 1.6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: #18181b;
        border: 1px solid rgba(239, 68, 68, 0.16);
        border-radius: 16px;
        color: #fca5a5;
        padding: 10px 18px;
        box-shadow: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%) !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-testid="stTabs"] div[role="tablist"] {
        border-bottom: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-grid">
            <div>
                <h1>MediScope Symptom Checker</h1>
                <p>Review symptom patterns, surface the most likely condition, and move through a healthcare website designed to feel structured, informative, and visually polished.</p>
                <div style="margin-top:18px;">
                    <span class="feature-pill">{len(encoder.classes_)} diseases</span>
                    <span class="feature-pill">{len(SYMPTOMS)} symptoms</span>
                    <span class="feature-pill">{len(dataset)} structured cases</span>
                    <span class="feature-pill">Prediction-focused interface</span>
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.14); border-radius:24px; padding:20px 22px;">
                <div style="font-size:0.86rem; opacity:0.9; text-transform:uppercase; letter-spacing:0.06em;">System profile</div>
                <div style="font-size:1.6rem; font-weight:800; margin-top:6px;">Condition intelligence view</div>
                <div style="margin-top:10px; opacity:0.92; line-height:1.65;">Built around broader symptom coverage, clearer ranking output, and focused pages for home, prediction, insights, and platform information.</div>
                <div style="margin-top:16px; display:grid; gap:10px;">
                    <div style="padding:12px 14px; border-radius:18px; background:rgba(0,0,0,0.22); border:1px solid rgba(255,255,255,0.1);">Respiratory, cardiac, digestive, infectious, endocrine, and neurological conditions</div>
                    <div style="padding:12px 14px; border-radius:18px; background:rgba(0,0,0,0.22); border:1px solid rgba(255,255,255,0.1);">Separate pages for prediction guidance, clinical insights, and platform information</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.empty()

tab_home, tab_predict, tab_insights, tab_about = st.tabs(["Home", "Prediction", "Insights", "About"])

with tab_home:
    st.markdown('<div class="page-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-copy">A structured symptom-based healthcare website with broader disease coverage, richer symptom input, and dedicated pages for prediction, insights, and platform information.</div>',
        unsafe_allow_html=True,
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.markdown(
            f'<div class="mini-stat"><h3>{len(dataset)}</h3><p>Total Cases</p></div>',
            unsafe_allow_html=True,
        )
    with summary_col2:
        st.markdown(
            f'<div class="mini-stat"><h3>{len(encoder.classes_)}</h3><p>Conditions Covered</p></div>',
            unsafe_allow_html=True,
        )
    with summary_col3:
        st.markdown(
            f'<div class="mini-stat"><h3>{len(SYMPTOMS)}</h3><p>Symptoms Available</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-box">
                <h4>Condition Coverage</h4>
                <p>The platform spans respiratory, cardiovascular, infectious, digestive, endocrine, dermatological, renal, and neurological disease groups to create a broader prediction space.</p>
            </div>
            <div class="feature-box">
                <h4>Symptom Specificity</h4>
                <p>Beyond common symptoms such as fever or fatigue, the system also models more discriminative signals like chest tightness, blood in urine, cough with mucus, neck stiffness, and rash blisters.</p>
            </div>
            <div class="feature-box">
                <h4>Website Structure</h4>
                <p>The home page now leads into dedicated website pages for clinical prediction, model insights, and platform information with a cleaner flow.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_predict:
    st.markdown(
        """
        <div class="page-hero">
            <div class="page-panel">
                <div class="page-title" style="margin-bottom:0;">Prediction Studio</div>
                <div class="page-copy" style="margin-top:8px; margin-bottom:0;">Select symptoms directly on the page, then review the main match and the ranked score profile.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_symptoms = st.multiselect(
        "Select observed symptoms",
        options=SYMPTOMS,
        format_func=lambda symptom: SYMPTOM_LABELS[symptom],
        placeholder="Choose symptoms here...",
    )
    predict_clicked = st.button("Predict Disease", use_container_width=True)

    left, right = st.columns([1.12, 0.88])
    with left:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0; color:#ffffff;">Selected Symptoms</h3>
                <p style="color:#d4d4d8; margin-bottom:0;">Build the patient profile from this section and review the ranked disease response below.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if selected_symptoms:
            selected_symptom_html = "".join(
                [f'<span class="selected-chip">{SYMPTOM_LABELS[symptom]}</span>' for symptom in selected_symptoms]
            )
            st.markdown(f'<div class="selected-chip-wrap">{selected_symptom_html}</div>', unsafe_allow_html=True)
        else:
            st.info("Choose one or more symptoms from the list above.")

        if predict_clicked:
            input_df = pd.DataFrame([{symptom: int(symptom in selected_symptoms) for symptom in SYMPTOMS}])
            prediction_encoded = model.predict(input_df)[0]
            prediction = encoder.inverse_transform([prediction_encoded])[0]
            probabilities = model.predict_proba(input_df)[0]
            top_indices = probabilities.argsort()[::-1][:5]
            top_probability = float(probabilities[top_indices[0]]) * 100
            theme_label, theme_bg, theme_fg = CONDITION_THEME.get(
                prediction, ("Predicted Profile", "#fce7f3", "#9d174d")
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Most likely disease</div>
                    <p class="result-value">{prediction}</p>
                    <div class="result-label">Confidence score: {top_probability:.2f}%</div>
                    <div class="result-label">{DISEASE_INFO.get(prediction, "Predicted from the selected symptom pattern.")}</div>
                    <div style="margin-top:14px;">
                        <span style="background:{theme_bg}; color:{theme_fg}; padding:9px 14px; border-radius:999px; font-size:0.9rem; font-weight:700;">{theme_label}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Prediction Breakdown")
            score_left, score_right = st.columns([1.05, 0.95])
            with score_left:
                st.markdown(
                    f"""
                    <div class="score-card">
                        <div class="result-label">Primary Match</div>
                        <div class="score-big">{prediction}</div>
                        <div class="section-note">{DISEASE_INFO.get(prediction, "Predicted from the selected symptom pattern.")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="score-card">
                        <div class="result-label">Confidence</div>
                        <div class="score-big">{top_probability:.2f}%</div>
                        <div class="section-note">This score reflects how strongly the selected symptom pattern matches the leading disease class in the current model.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with score_right:
                st.markdown("##### Top Condition Scores")
                score_colors = ["#dc2626", "#ef4444", "#f97316", "#7c2d12", "#991b1b"]
                for idx, disease_index in enumerate(top_indices):
                    disease_name = encoder.inverse_transform([disease_index])[0]
                    render_score_bar(
                        disease_name,
                        float(probabilities[disease_index]) * 100,
                        score_colors[idx % len(score_colors)],
                    )

            probability_table = pd.DataFrame(
                {
                    "Disease": encoder.inverse_transform(top_indices),
                    "Probability (%)": [round(float(probabilities[i]) * 100, 2) for i in top_indices],
                }
            )
            st.markdown("#### Ranked Conditions")
            st.dataframe(probability_table, width="stretch", hide_index=True)
            st.caption("Use this output as a clinical-style prediction summary, not as a medical diagnosis.")
        else:
            st.markdown(
                """
                <div class="spotlight">
                    <div style="font-size:0.9rem; color:#fca5a5; font-weight:700;">Prediction Canvas</div>
                    <div style="font-size:1.5rem; color:#ffffff; font-weight:800; margin-top:6px;">Ready for symptom analysis</div>
                    <div style="color:#d4d4d8; margin-top:8px;">Once you choose symptoms and press Predict Disease, this area will transform into the diagnostic result card with ranked conditions.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#ffffff;">Case Builder Notes</h3>
                <p style="color:#d4d4d8; margin-bottom:10px;">Use disease-specific symptoms to make the output look more realistic. Examples: chest tightness for respiratory disease, neck stiffness for meningitis, blood in urine for kidney or urinary disease.</p>
                <div class="disease-cloud">
                    <span class="disease-chip">Chest Tightness</span>
                    <span class="disease-chip">Cough With Mucus</span>
                    <span class="disease-chip">Neck Stiffness</span>
                    <span class="disease-chip">Blood In Urine</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#12355b;">Prediction Quality</h3>
                <p class="section-note">For clearer results, combine core symptoms with one or two support symptoms rather than selecting every symptom at once. This usually creates a cleaner disease ranking and a stronger main result card.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_insights:
    st.markdown(
        f"""
            <div class="page-hero">
                <div class="insight-banner">
                    <div style="font-size:0.92rem; opacity:0.92;">Clinical Insights</div>
                    <div style="font-size:2rem; font-weight:800; margin-top:6px;">How this prediction system is structured</div>
                    <div style="margin-top:10px; max-width:760px; opacity:0.96;">This page explains the model, symptom coverage, and how the platform interprets symptom combinations across a broad condition library.</div>
                </div>
                <div class="page-panel">
                    <div style="font-size:0.86rem; color:#fca5a5; font-weight:700;">Snapshot</div>
                    <div style="font-size:1.2rem; color:#ffffff; font-weight:800; margin-top:6px;">{metrics['accuracy'] * 100:.2f}% measured accuracy</div>
                    <div style="color:#d4d4d8; margin-top:8px;">Measured on the generated dataset with broader symptom specificity and disease coverage.</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="insight-grid">
            <div class="metric-panel">
                <h4>Prediction Engine</h4>
                <p>Algorithm: Random Forest Classifier</p>
                <p>Total training cases: {len(dataset)}</p>
                <p>Diseases covered: {len(encoder.classes_)}</p>
                <p>Symptoms modeled: {len(SYMPTOMS)}</p>
            </div>
            <div class="metric-panel">
                <h4>Coverage Summary</h4>
                <p>Includes respiratory, infectious, digestive, urinary, skin, chronic, thyroid, cardiac, and neurological conditions.</p>
                <p>Expanded with more symptom-specific patterns for globally important diseases.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    insights_left, insights_right = st.columns(2)
    with insights_left:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#ffffff;">How The Model Interprets Symptoms</h3>
                <p class="section-note">Each selected symptom becomes a binary feature in the input vector. The classifier compares that pattern against thousands of simulated disease cases and estimates the nearest disease matches. Diseases with more distinct signatures, such as meningitis or appendicitis, often produce sharper score separation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insights_right:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#ffffff;">Why More Specific Symptoms Matter</h3>
                <p class="section-note">Generic symptoms like fever and fatigue occur across many diseases, so they do not separate classes very well on their own. More specific symptoms such as neck stiffness, blood in urine, cough with mucus, chest tightness, and rash blisters help the model narrow the prediction more effectively.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="info-card">
            <h3 style="margin-top:0; color:#ffffff;">Interpretation Notes</h3>
            <p class="section-note">The score shown in the prediction page is a model confidence estimate within the current environment, not a clinical certainty score. The platform is designed to present classification workflow, symptom reasoning, and healthcare interface quality in a clear and accessible format.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_about:
    st.markdown(
        """
            <div class="page-hero">
                <div class="insight-banner">
                    <div style="font-size:0.92rem; opacity:0.92;">About MediScope</div>
                    <div style="font-size:2rem; font-weight:800; margin-top:6px;">A classification-based disease prediction platform</div>
                    <div style="margin-top:10px; max-width:760px; opacity:0.96;">MediScope combines machine learning with a modern healthcare-style interface to create a disease prediction experience that is structured, informative, and visually refined.</div>
                </div>
                <div class="page-panel">
                    <div style="font-size:0.86rem; color:#fca5a5; font-weight:700;">Platform Positioning</div>
                    <div style="font-size:1.2rem; color:#ffffff; font-weight:800; margin-top:6px;">Built as a polished healthcare web experience</div>
                    <div style="color:#d4d4d8; margin-top:8px;">The focus is not only prediction, but also a stronger website experience with cleaner information flow and richer symptom coverage.</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-box">
                <h4>Clinical Focus</h4>
                <p>Support symptom-based condition screening through a polished interface that feels like a real healthcare website.</p>
            </div>
            <div class="feature-box">
                <h4>Prediction Engine</h4>
                <p>The app uses classification to map patient symptom patterns to the most likely disease classes and rank the top matches.</p>
            </div>
            <div class="feature-box">
                <h4>Website Experience</h4>
                <p>The interface uses a cleaner webpage flow with in-page section selection rather than sidebar navigation.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    about_left, about_right = st.columns(2)
    with about_left:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#ffffff;">Why MediScope</h3>
                <p class="section-note">MediScope brings together a disease classification engine and a healthcare-style presentation layer in one platform. It is designed to make symptom-driven screening feel clearer, more structured, and easier to interpret.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with about_right:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="margin-top:0; color:#ffffff;">Service Scope</h3>
                <p class="section-note">The current experience focuses on symptom-based disease ranking across a wide set of globally relevant conditions. It emphasizes readability, ranking clarity, and page-level navigation rather than patient record management.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="info-card">
            <h3 style="margin-top:0; color:#ffffff;">Service Roadmap</h3>
            <div class="timeline">
                <div class="timeline-item">
                    <h5>1. Expanded Clinical Data</h5>
                    <p>Replace synthetic patterns with a real disease-symptom dataset or a curated CSV source for stronger realism.</p>
                </div>
                <div class="timeline-item">
                    <h5>2. Report Generation</h5>
                    <p>Add downloadable patient summary cards or PDF-style reports for stronger reporting and clinical-style presentation.</p>
                </div>
                <div class="timeline-item">
                    <h5>3. Role-Based Experience</h5>
                    <p>Introduce patient, admin, or doctor-facing views to expand the platform into a more complete digital health experience.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
