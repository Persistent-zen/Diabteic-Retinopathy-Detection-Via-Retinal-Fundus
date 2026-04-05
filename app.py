import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from predict import predict_image, generate_gradcam
import datetime


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="DR Screening ",
    layout="wide"
)

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
}

div[data-testid="stMetric"] {
    background-color: white;
    border-radius: 12px;
    padding: 18px;
    border: 1px solid #E5EDF5;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
}

hr {
    border: none;
    border-top: 1px solid #E5EDF5;
}

</style>
""", unsafe_allow_html=True)

# ---------------- CUSTOM NAVBAR ----------------

st.markdown("""
<style>

.navbar {
    background-color: #ffffff;
    padding: 14px 24px;
    border-bottom: 2px solid #e6eef7;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.navbar-title {
    font-size: 26px;
    font-weight: 600;
    color: #1f77b4;
}

.section-header {
    font-size: 22px;
    font-weight: 600;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="navbar">
<div class="navbar-title">
🩺 DR Screening Assistant
</div>
<div>
Retinal Analysis Dashboard
</div>
</div>
""", unsafe_allow_html=True)


# ---------------- CONSTANTS ----------------

labels = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


interpretation = {

"No DR":
"No visible signs of diabetic retinopathy detected.",

"Mild":
"Microaneurysms detected. Monitoring recommended.",

"Moderate":
"Hemorrhages/exudates suspected. Referral advised.",

"Severe":
"High progression risk. Urgent ophthalmic review recommended.",

"Proliferative DR":
"Advanced neovascularization suspected. Immediate intervention required."
}


risk_map = {

"No DR": "Low",
"Mild": "Low",
"Moderate": "Medium",
"Severe": "High",
"Proliferative DR": "Critical"
}


severity_icon = {

"No DR": "🟢",
"Mild": "🟡",
"Moderate": "🟠",
"Severe": "🔴",
"Proliferative DR": "🚨"
}


# ---------------- SIDEBAR NAVIGATION ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "Screening Dashboard",
        "Explainability Insights",
        "Patient Education",
        "System Information"
    ]
)


st.sidebar.markdown("---")

st.sidebar.markdown("""
### Model Information

Architecture: MobileNetV2  
Input Size: 224 × 224  
Classes: 5-stage DR severity  
Explainability: Grad-CAM  

⚠ Research prototype
""")


# ---------------- IMAGE QUALITY CHECK ----------------

def check_image_quality(img):

    if img.std() < 25:
        return False

    return True


# ---------------- PAGE 1 : SCREENING DASHBOARD ----------------

if page == "Screening Dashboard":

    st.markdown("## Upload Retinal Fundus Image")

    uploaded_files = st.file_uploader(
        "Upload retinal image(s)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )


    if uploaded_files:

        for uploaded_file in uploaded_files:

            st.divider()

            st.subheader(f"Processing: {uploaded_file.name}")

            image = Image.open(uploaded_file)

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )


            temp_path = "temp.jpg"

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())


            prediction, original_img, img_input = predict_image(temp_path)


            predicted_class_index = np.argmax(prediction)

            predicted_label = labels[predicted_class_index]

            confidence = prediction[0][predicted_class_index] * 100


            # QUALITY WARNINGS

            if not check_image_quality(original_img):

                st.warning("Low image quality detected.")


            if confidence < 50:

                st.warning("Low confidence prediction. Clinical review advised.")


            # CLINICAL SUMMARY CARDS

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Diagnosis",
                f"{severity_icon[predicted_label]} {predicted_label}"
            )

            col2.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            col3.metric(
                "Risk Level",
                risk_map[predicted_label]
            )


            # INTERPRETATION PANEL

            st.info(interpretation[predicted_label])


            # CONFIDENCE DISTRIBUTION

            st.subheader("Prediction Confidence Distribution")

            confidence_df = pd.DataFrame(
                prediction[0],
                index=labels,
                columns=["Confidence"]
            )

            st.bar_chart(confidence_df)


            # SAVE STATE FOR INSIGHTS TAB

            st.session_state["last_prediction"] = prediction
            st.session_state["last_original"] = original_img
            st.session_state["last_input"] = img_input
            st.session_state["last_uploaded_image"] = image
            st.session_state["last_filename"] = uploaded_file.name
            st.session_state["last_label"] = predicted_label
            st.session_state["last_confidence"] = confidence


            # REPORT DOWNLOAD

            report = f"""

Diabetic Retinopathy Screening Report

Image: {uploaded_file.name}

Prediction: {predicted_label}

Confidence: {confidence:.2f}%

Risk Level: {risk_map[predicted_label]}

Timestamp: {datetime.datetime.now()}

"""

            st.download_button(
                "Download Screening Report",
                report,
                file_name="DR_report.txt"
            )


# ---------------- PAGE 2 : EXPLAINABILITY ----------------

elif page == "Explainability Insights":

    st.markdown("## Grad-CAM Explainability Visualization")

    if "last_input" not in st.session_state:

        st.warning("Upload an image first in Screening Dashboard.")

    else:

        heatmap, overlay = generate_gradcam(
            st.session_state["last_input"],
            st.session_state["last_original"]
        )


        st.markdown("""
Highlighted regions indicate retinal structures influencing
the model’s prediction such as:

• microaneurysms  
• hemorrhages  
• exudates  
• vascular abnormalities
""")


        col1, col2, col3 = st.columns(3)


        with col1:

            st.image(
                st.session_state["last_uploaded_image"],
                caption="Original",
                use_container_width=True
            )


        with col2:

            st.image(
                cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
                caption="Heatmap",
                use_container_width=True
            )


        with col3:

            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                caption="Overlay",
                use_container_width=True
            )


        st.subheader("Confidence Distribution")

        confidence_df = pd.DataFrame(
            st.session_state["last_prediction"][0],
            index=labels,
            columns=["Confidence"]
        )

        st.bar_chart(confidence_df)


elif page == "Patient Education":

    st.markdown("## Understanding Diabetic Retinopathy")

    st.markdown("""
Diabetic Retinopathy is a complication of diabetes that damages the small
blood vessels inside the retina (the light-sensitive tissue at the back of the eye).

Early detection helps prevent permanent vision loss.
""")

    st.markdown("### Common retinal findings detected by AI")

    st.markdown("""
**Microaneurysms**

Tiny swellings in retinal blood vessels.
Usually the earliest visible sign of diabetic retinopathy.

**Hemorrhages**

Small bleeding spots caused by fragile vessels leaking blood.

**Hard Exudates**

Fat deposits leaking from damaged vessels.

**Neovascularization**

Growth of abnormal fragile blood vessels seen in advanced disease.
""")


    st.markdown("### What each severity level means")

    severity_help = {

        "No DR":
        "No visible retinal damage detected. Continue routine annual screening.",

        "Mild":
        "Early-stage vessel swelling detected. Usually manageable with monitoring.",

        "Moderate":
        "Blood vessel leakage increasing. Consultation with an ophthalmologist recommended.",

        "Severe":
        "Multiple blocked vessels reducing oxygen supply to retina. Urgent specialist review needed.",

        "Proliferative DR":
        "Advanced disease stage with abnormal vessel growth. Immediate treatment evaluation required."
    }

    for stage, explanation in severity_help.items():

        st.markdown(f"**{stage}**")

        st.write(explanation)


    st.markdown("### Recommended follow-up actions")

    next_steps = {

        "No DR":
        "Schedule routine screening once every 12 months.",

        "Mild":
        "Follow-up screening recommended within 6–12 months.",

        "Moderate":
        "Consult ophthalmologist within 3–6 months.",

        "Severe":
        "Consult retina specialist urgently.",

        "Proliferative DR":
        "Immediate specialist consultation required."
    }

    for stage, step in next_steps.items():

        st.markdown(f"**{stage} →** {step}")


    st.markdown("### Understanding the heatmap visualization")

    st.info("""
The heatmap highlights retinal areas the AI system focused on when making its decision.

Red regions indicate structures that influenced the prediction.

This does NOT directly represent disease severity.
It only shows areas important for the model’s analysis.
""")


    st.markdown("### When should you consult a doctor immediately?")

    st.markdown("""
Seek urgent medical attention if you experience:

• sudden vision loss  
• blurred vision worsening rapidly  
• floating spots in vision  
• dark areas in your field of view
""")


    st.markdown("### Trusted medical resources")

    st.markdown("""
American Academy of Ophthalmology  
https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy

National Eye Institute  
https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy

NHS Guide  
https://www.nhs.uk/conditions/diabetic-retinopathy/
""")


# ---------------- PAGE 3 : SYSTEM INFORMATION ----------------

elif page == "System Information":

    st.markdown("## System Overview")

    st.markdown("""
This system assists diabetic retinopathy screening using deep learning.

Capabilities:

• Multi-class severity detection  
• Grad-CAM explainability  
• Confidence distribution analytics  
• Batch screening workflow  

Architecture:

MobileNetV2 Transfer Learning Pipeline

Limitations:

This tool supports screening workflows it  
is not a substitute for a medical professional. 
""")


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption("""
diabetic retinopathy screening prototype  
Built using transfer learning + explainable AI
""")