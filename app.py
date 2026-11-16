import sqlite3
from pathlib import Path
#comment meghan_test_branch


import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------

st.set_page_config(
    page_title="MediCare Multispecialty Hospital",
    page_icon="🏥",
    layout="wide"
)


# ------------------------------------------------
# DASHBOARD COLOUR AND CONTRAST SETTINGS
# ------------------------------------------------

st.markdown(
    """
    <style>
    
    [data-testid="stAppViewContainer"] {
        background-color: #F4F7FB;
    }
    
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3 {
        color: #073B4C !important;
    }
    
    [data-testid="stAppViewContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"] {
        color: #1F2937 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #0077B6 !important;
        font-size: 28px;
        font-weight: 700;
    }
    
    button[data-baseweb="tab"] {
        color: #17324D !important;
        font-weight: 700;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0077B6 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #073B4C;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------
# DATABASE CONNECTION
# ------------------------------------------------

DATABASE_PATH = (
    Path(__file__).parent / "hospital.db"
)


@st.cache_data
def load_hospital_data():
    
    connection = sqlite3.connect(
        DATABASE_PATH
    )
    
    try:
        
        patients = pd.read_sql(
            "SELECT * FROM Patients",
            connection
        )
        
        doctors = pd.read_sql(
            "SELECT * FROM Doctors",
            connection
        )
        
        appointments = pd.read_sql(
            "SELECT * FROM Appointments",
            connection
        )
        
        treatments = pd.read_sql(
            "SELECT * FROM Treatments",
            connection
        )
        
        beds = pd.read_sql(
            "SELECT * FROM BedAllocation",
            connection
        )
        
        pharmacy = pd.read_sql(
            "SELECT * FROM PharmacySales",
            connection
        )
        
        surveillance = pd.read_sql(
            "SELECT * FROM DiseaseSurveillance",
            connection
        )
        
        return (
            patients,
            doctors,
            appointments,
            treatments,
            beds,
            pharmacy,
            surveillance
        )
    
    finally:
        
        connection.close()


if not DATABASE_PATH.exists():
    
    st.error(
        "hospital.db was not found. "
        "Keep hospital.db in the same folder as app.py."
    )
    
    st.stop()


(
    patients,
    doctors,
    appointments,
    treatments,
    beds,
    pharmacy,
    surveillance
) = load_hospital_data()


# ------------------------------------------------
# DATE CONVERSION
# ------------------------------------------------

patients["AdmissionDate"] = pd.to_datetime(
    patients["AdmissionDate"]
)

appointments["VisitDate"] = pd.to_datetime(
    appointments["VisitDate"]
)

treatments["TreatmentDate"] = pd.to_datetime(
    treatments["TreatmentDate"]
)

beds["AdmissionDate"] = pd.to_datetime(
    beds["AdmissionDate"]
)

beds["DischargeDate"] = pd.to_datetime(
    beds["DischargeDate"]
)

pharmacy["SaleDate"] = pd.to_datetime(
    pharmacy["SaleDate"]
)

surveillance["Date"] = pd.to_datetime(
    surveillance["Date"]
)


# ------------------------------------------------
# PATIENT SUMMARY AND RISK SEGMENTATION
# ------------------------------------------------

treatment_summary = (
    treatments
    .groupby(
        "PatientID",
        as_index=False
    )
    .agg(
        TotalTreatmentCost=(
            "Cost",
            "sum"
        )
    )
)


patient_summary = patients.merge(
    treatment_summary,
    on="PatientID",
    how="left"
)


patient_summary[
    "TotalTreatmentCost"
] = patient_summary[
    "TotalTreatmentCost"
].fillna(0)


patient_summary["Segment"] = pd.cut(
    patient_summary[
        "TotalTreatmentCost"
    ],
    
    bins=[
        -1,
        25000,
        75000,
        np.inf
    ],
    
    labels=[
        "Standard",
        "High Risk",
        "Critical"
    ]
)


# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title(
    "🏥 MediCare Multispecialty Hospital"
)

st.subheader(
    "Hospital Analytics & Predictive Care Platform"
)

st.caption(
    "Interactive dashboard for patient risk, "
    "doctor performance, bed allocation, "
    "disease outbreaks, medicine demand, "
    "and department revenue."
)


# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------

st.sidebar.header("Dashboard Filters")


department_options = sorted(
    doctors["Department"].unique()
)

selected_departments = st.sidebar.multiselect(
    "Select Department",
    options=department_options,
    default=department_options
)


city_options = sorted(
    patients["City"].unique()
)

selected_cities = st.sidebar.multiselect(
    "Select Patient City",
    options=city_options,
    default=city_options
)


minimum_age = int(
    patients["Age"].min()
)

maximum_age = int(
    patients["Age"].max()
)

selected_age_range = st.sidebar.slider(
    "Select Patient Age Range",
    min_value=minimum_age,
    max_value=maximum_age,
    value=(
        minimum_age,
        maximum_age
    )
)


minimum_cost = int(
    patient_summary[
        "TotalTreatmentCost"
    ].min()
)

maximum_cost = int(
    patient_summary[
        "TotalTreatmentCost"
    ].max()
)

selected_cost_range = st.sidebar.slider(
    "Select Treatment Cost Range",
    min_value=minimum_cost,
    max_value=maximum_cost,
    value=(
        minimum_cost,
        maximum_cost
    ),
    step=5000
)


minimum_date = patients[
    "AdmissionDate"
].min().date()

maximum_date = patients[
    "AdmissionDate"
].max().date()

selected_date_range = st.sidebar.slider(
    "Select Admission Date Range",
    min_value=minimum_date,
    max_value=maximum_date,
    value=(
        minimum_date,
        maximum_date
    )
)


category_options = sorted(
    pharmacy["Category"].unique()
)

selected_categories = st.sidebar.multiselect(
    "Select Medicine Category",
    options=category_options,
    default=category_options
)


# ------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------

filtered_patients = patient_summary[
    (
        patient_summary["City"].isin(
            selected_cities
        )
    )
    &
    (
        patient_summary["Age"].between(
            selected_age_range[0],
            selected_age_range[1]
        )
    )
    &
    (
        patient_summary[
            "TotalTreatmentCost"
        ].between(
            selected_cost_range[0],
            selected_cost_range[1]
        )
    )
    &
    (
        patient_summary["AdmissionDate"].dt.date
        >= selected_date_range[0]
    )
    &
    (
        patient_summary["AdmissionDate"].dt.date
        <= selected_date_range[1]
    )
].copy()


if filtered_patients.empty:
    
    st.warning(
        "No patient records match the selected filters. "
        "Please adjust your filter values."
    )
    
    st.stop()


filtered_patient_ids = set(
    filtered_patients["PatientID"]
)


filtered_doctors = doctors[
    doctors["Department"].isin(
        selected_departments
    )
].copy()


filtered_doctor_ids = set(
    filtered_doctors["DoctorID"]
)


filtered_appointments = appointments[
    (
        appointments["PatientID"].isin(
            filtered_patient_ids
        )
    )
    &
    (
        appointments["DoctorID"].isin(
            filtered_doctor_ids
        )
    )
].copy()


filtered_treatments = treatments[
    treatments["PatientID"].isin(
        filtered_patient_ids
    )
].copy()


filtered_beds = beds[
    beds["PatientID"].isin(
        filtered_patient_ids
    )
].copy()


filtered_pharmacy = pharmacy[
    pharmacy["Category"].isin(
        selected_categories
    )
].copy()


filtered_surveillance = surveillance[
    surveillance["City"].isin(
        selected_cities
    )
].copy()


# ------------------------------------------------
# KPI CARDS
# ------------------------------------------------

consultation_revenue = filtered_appointments[
    "ConsultationFee"
].sum()

treatment_revenue = filtered_treatments[
    "Cost"
].sum()

total_revenue = (
    consultation_revenue
    + treatment_revenue
)


high_risk_count = len(
    filtered_patients[
        filtered_patients["Segment"].isin(
            [
                "High Risk",
                "Critical"
            ]
        )
    ]
)


column1, column2, column3, column4 = st.columns(4)

column1.metric(
    "Filtered Patients",
    filtered_patients["PatientID"].nunique()
)

column2.metric(
    "Appointments",
    filtered_appointments[
        "AppointmentID"
    ].nunique()
)

column3.metric(
    "High-Risk Patients",
    high_risk_count
)

column4.metric(
    "Total Revenue",
    f"₹{total_revenue:,.0f}"
)


# ------------------------------------------------
# CREATE TABS
# ------------------------------------------------

overview_tab, patient_tab, operations_tab = st.tabs(
    [
        "📊 Overview",
        "🧑‍⚕️ Patients & Risk",
        "🏥 Operations & Forecast"
    ]
)


# ------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------

with overview_tab:
    
    st.header("Doctor and Department Performance")
    
    
    doctor_revenue = (
        filtered_doctors
        .merge(
            filtered_appointments,
            on="DoctorID",
            how="left"
        )
        .groupby(
            [
                "DoctorName",
                "Department"
            ],
            as_index=False
        )
        .agg(
            Revenue=(
                "ConsultationFee",
                "sum"
            ),
            
            AppointmentCount=(
                "AppointmentID",
                "count"
            )
        )
        .sort_values(
            "Revenue",
            ascending=False
        )
    )
    
    
    left_column, right_column = st.columns(2)
    
    
    with left_column:
        
        doctor_chart = px.bar(
            doctor_revenue,
            x="DoctorName",
            y="Revenue",
            color="Department",
            title="Top Revenue Doctors",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        doctor_chart.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#17324D"
        )
        
        st.plotly_chart(
            doctor_chart,
            use_container_width=True
        )
    
    
    with right_column:
        
        department_revenue = (
            doctor_revenue
            .groupby(
                "Department",
                as_index=False
            )
            .agg(
                Revenue=(
                    "Revenue",
                    "sum"
                )
            )
            .sort_values(
                "Revenue",
                ascending=False
            )
        )
        
        
        department_chart = px.pie(
            department_revenue,
            names="Department",
            values="Revenue",
            title="Department Revenue Analysis",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        department_chart.update_layout(
            font_color="#17324D"
        )
        
        st.plotly_chart(
            department_chart,
            use_container_width=True
        )
    
    
    st.subheader("Doctor Performance Table")
    
    st.dataframe(
        doctor_revenue,
        use_container_width=True
    )


# ------------------------------------------------
# TAB 2: PATIENTS AND RISK
# ------------------------------------------------

with patient_tab:
    
    st.header("Patient Risk Analysis")
    
    
    left_column, right_column = st.columns(2)
    
    
    with left_column:
        
        segment_count = (
            filtered_patients
            .groupby(
                "Segment",
                observed=False
            )
            .size()
            .reset_index(
                name="PatientCount"
            )
        )
        
        
        segment_chart = px.bar(
            segment_count,
            x="Segment",
            y="PatientCount",
            color="Segment",
            title="Patient Risk Segments",
            color_discrete_map={
                "Standard": "#2A9D8F",
                "High Risk": "#F4A261",
                "Critical": "#E63946"
            }
        )
        
        segment_chart.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#17324D"
        )
        
        st.plotly_chart(
            segment_chart,
            use_container_width=True
        )
    
    
    with right_column:
        
        age_chart = px.scatter(
            filtered_patients,
            x="Age",
            y="TotalTreatmentCost",
            color="Segment",
            hover_name="PatientName",
            title="Age vs Treatment Cost",
            color_discrete_map={
                "Standard": "#2A9D8F",
                "High Risk": "#F4A261",
                "Critical": "#E63946"
            }
        )
        
        age_chart.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#17324D"
        )
        
        st.plotly_chart(
            age_chart,
            use_container_width=True
        )
    
    
    st.subheader("High-Risk Patients")
    
    
    high_risk_patients = filtered_patients[
        filtered_patients["Segment"].isin(
            [
                "High Risk",
                "Critical"
            ]
        )
    ]
    
    
    st.dataframe(
        high_risk_patients[
            [
                "PatientID",
                "PatientName",
                "Age",
                "City",
                "TotalTreatmentCost",
                "Segment"
            ]
        ],
        use_container_width=True
    )


# ------------------------------------------------
# TAB 3: OPERATIONS AND FORECAST
# ------------------------------------------------

with operations_tab:
    
    st.header("Bed, Disease and Medicine Analysis")
    
    
    occupancy_dates = []
    
    
    for _, row in filtered_beds.iterrows():
        
        last_occupied_date = (
            row["DischargeDate"]
            - pd.Timedelta(days=1)
        )
        
        
        dates = pd.date_range(
            start=row["AdmissionDate"],
            end=last_occupied_date,
            freq="D"
        )
        
        
        occupancy_dates.extend(dates)
    
    
    if occupancy_dates:
        
        daily_bed_occupancy = (
            pd.Series(occupancy_dates)
            .value_counts()
            .sort_index()
            .rename("OccupiedBeds")
            .reset_index()
        )
        
        daily_bed_occupancy.columns = [
            "Date",
            "OccupiedBeds"
        ]
        
        
        bed_chart = px.line(
            daily_bed_occupancy,
            x="Date",
            y="OccupiedBeds",
            markers=True,
            title="Bed Occupancy Trends",
            color_discrete_sequence=["#0077B6"]
        )
        
        bed_chart.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#17324D"
        )
        
        st.plotly_chart(
            bed_chart,
            use_container_width=True
        )
    
    else:
        
        st.info(
            "No bed-allocation data is available "
            "for the selected patient filters."
        )
    
    
    st.caption(
        "The dataset does not provide total bed capacity, "
        "so this chart shows allocated beds per day."
    )
    
    
    left_column, right_column = st.columns(2)
    
    
    with left_column:
        
        monthly_admissions = (
            filtered_patients
            .set_index("AdmissionDate")
            .resample("MS")
            .size()
            .rename("Admissions")
            .reset_index()
        )
        
        
        admissions_chart = px.line(
            monthly_admissions,
            x="AdmissionDate",
            y="Admissions",
            markers=True,
            title="Monthly Admissions",
            color_discrete_sequence=["#2A9D8F"]
        )
        
        admissions_chart.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#17324D"
        )
        
        st.plotly_chart(
            admissions_chart,
            use_container_width=True
        )
    
    
    with right_column:
        
        outbreak_chart = px.scatter(
            filtered_surveillance,
            x="AQI",
            y="FluCases",
            color="City",
            size="FluCases",
            title="Disease Outbreak Analysis",
            color_discrete_sequence=px.colors.qualitative.Dark2
        )
        
        outbreak_chart.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#17324D"
        )
        
        st.plotly_chart(
            outbreak_chart,
            use_container_width=True
        )
    
    
    # Medicine demand
    
    medicine_demand = (
        filtered_pharmacy
        .groupby(
            [
                "MedicineName",
                "Category"
            ],
            as_index=False
        )
        .agg(
            QuantitySold=(
                "Quantity",
                "sum"
            )
        )
        .sort_values(
            "QuantitySold",
            ascending=False
        )
    )
    
    
    daily_medicine_demand = (
        filtered_pharmacy
        .set_index("SaleDate")[
            "Quantity"
        ]
        .resample("D")
        .sum()
    )
    
    
    if len(daily_medicine_demand) > 0:
        
        next_day_forecast = round(
            daily_medicine_demand.tail(3).mean(),
            2
        )
    
    else:
        
        next_day_forecast = 0
    
    
    medicine_chart = px.bar(
        medicine_demand,
        x="MedicineName",
        y="QuantitySold",
        color="Category",
        title="Medicine Demand by Product",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    medicine_chart.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#17324D"
    )
    
    st.plotly_chart(
        medicine_chart,
        use_container_width=True
    )
    
    
    st.metric(
        "Simple Next-Day Medicine Demand Forecast",
        next_day_forecast
    )
    
    
    st.caption(
        "Forecast = average quantity sold during "
        "the latest three days. More historical data "
        "is needed for a reliable forecast."
    )