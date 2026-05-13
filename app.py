import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

# إعداد واجهة المستخدم
st.set_page_config(page_title="NeuroVision MRI Analysis", layout="wide")

st.title("🧠 NeuroVision: MRI Condition Classifier")
st.markdown("""
هذا التطبيق يستخدم الذكاء الاصطناعي للتنبؤ بالحالة الصحية بناءً على قياسات الرنين المغناطيسي (MRI).
""")

# --- قسم إدخال البيانات (Sidebar) ---
st.sidebar.header("📥 إدخال بيانات المريض")

def user_input_features():
    age = st.sidebar.slider("العمر (Age)", 10, 80, 30)
    gender = st.sidebar.selectbox("الجنس (Gender)", ("Male", "Female"))
    gray_matter = st.sidebar.number_input("Gray Matter Volume", 400.0, 800.0, 600.0)
    white_matter = st.sidebar.number_input("White Matter Volume", 300.0, 600.0, 430.0)
    brain_stem = st.sidebar.number_input("Brain Stem Size", 15.0, 35.0, 24.0)
    optic_nerve = st.sidebar.number_input("Optic Nerve Thickness", 2.0, 5.0, 3.7)
    corpus = st.sidebar.number_input("Corpus Thickness", 1.0, 5.0, 2.6)
    left_eye = st.sidebar.slider("Vision (Left Eye)", 0.0, 1.0, 0.8)
    right_eye = st.sidebar.slider("Vision (Right Eye)", 0.0, 1.0, 0.8)
    movement = st.sidebar.slider("Eye Movement Score", 0, 10, 5)
    melanin = st.sidebar.slider("Melanin Score", 1.0, 5.0, 3.0)
    pigment = st.sidebar.slider("Pigment Rate", 0, 100, 50)

    data = {
        'Age': age,
        'Gender': 0 if gender == "Male" else 1,
        'GrayMatter': gray_matter,
        'WhiteMatter': white_matter,
        'BrainStem': brain_stem,
        'OpticNerve': optic_nerve,
        'CorpusThickness': corpus,
        'LeftEyeVision': left_eye,
        'RightEyeVision': right_eye,
        'EyeMovementScore': movement,
        'MelaninScore': melanin,
        'PigmentRate': pigment
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- محاكاة تدريب النموذج (أو تحميل نموذج جاهز) ---
# ملاحظة: في المشاريع الحقيقية نستخدم نموذجاً تم تدريبه مسبقاً وحفظه بصيغة .pkl
@st.cache_resource
def train_model():
    # هنا نضع كود التدريب السريع من مشروعك (بيانات وهمية للتبسيط)
    X_train = np.random.rand(100, 12)
    y_train = np.random.randint(0, 3, 100)
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    return model

model = train_model()

# --- التنبؤ ---
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

classes = {0: "Healthy (سليم)", 1: "Albinism (مهق)", 2: "Achiasma (لا تصالبية)"}

# --- عرض النتائج ---
st.subheader("📊 نتيجة التحليل")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="الحالة المتوقعة", value=classes[prediction[0]])

with col2:
    st.write("### احتمالية التنبؤ")
    prob_df = pd.DataFrame(prediction_proba, columns=classes.values())
    st.bar_chart(prob_df.T)

st.write("---")
st.write("💡 **ملاحظة:** هذا التطبيق للأغراض التعليمية فقط ويعتمد على بيانات اصطناعية.")