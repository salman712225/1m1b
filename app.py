import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Aquasense | AI Water Intelligence",
    page_icon="💧",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv("aquasense_ai_output.csv")

df = load_data()

# ==================================================
# HERO
# ==================================================
st.markdown("""
# 💧Aquasense 
### *AI, ML & Data Science for Climate-Resilient Water Governance*

A decision-intelligence platform that transforms **water data into predictions,
risk signals, and policy-ready insights**.  
The dashboard speaks — you don’t have to.
""")

st.markdown("---")

# ==================================================
# SIDEBAR CONTROLS
# =================================================
st.sidebar.header("🌍 Controls")

country = st.sidebar.selectbox("Select Country", sorted(df["Country"].unique()))

year_range = st.sidebar.slider(
    "Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (2010, int(df["Year"].max()))
)

data = df[
    (df["Country"] == country) &
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1])
]

# ==================================================
# KPI SNAPSHOT
# ==================================================
st.subheader("📌 Executive Snapshot")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Avg Demand (BCM)",
          round(data["Total Water Consumption (Billion Cubic Meters)"].mean(), 2))
k2.metric("Per Capita Use (L/day)",
          round(data["Per Capita Water Use (Liters per Day)"].mean(), 1))
k3.metric("Groundwater Depletion (%)",
          round(data["Groundwater Depletion Rate (%)"].mean(), 1))
k4.metric("Dominant Risk Level",
          data["Risk_Level"].mode()[0])

st.markdown("---")

# ==================================================
# ROW 1 — TREND | FORECAST | CARBON
# ==================================================
st.subheader("📈 Demand, Forecast & Climate Impact")

c1, c2, c3 = st.columns(3)

with c1:
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(data["Year"], data["Total Water Consumption (Billion Cubic Meters)"], marker="o")
    ax.set_ylabel("BCM")
    st.pyplot(fig)
    st.caption("Water demand has increased steadily over time.")

with c2:
    X_lr = data[["Year"]]
    y_lr = data["Total Water Consumption (Billion Cubic Meters)"]

    lr = LinearRegression()
    lr.fit(X_lr, y_lr)

    future_years = np.array(range(data["Year"].max()+1, data["Year"].max()+6)).reshape(-1,1)
    forecast_lr = lr.predict(future_years)

    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(data["Year"], y_lr, label="Actual")
    ax.plot(future_years.flatten(), forecast_lr, linestyle="--", marker="x", label="Forecast")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    st.caption("AI predicts continued growth under current trends.")

with c3:
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(data["Year"], data["Estimated_CO2_Emissions_tons"], color="crimson", marker="x")
    ax.set_ylabel("CO₂ (tons)")
    st.pyplot(fig)
    st.caption("Water demand drives energy use and carbon emissions.")

st.markdown("---")

# ==================================================
# ROW 2 — MULTIVARIATE DRIVERS
# ==================================================
st.subheader("🔗 Multivariate Drivers of Water Stress")

c4, c5, c6 = st.columns(3)

with c4:
    fig, ax = plt.subplots(figsize=(4,3))
    sns.scatterplot(data=data,
                    x="Rainfall Impact (Annual Precipitation in mm)",
                    y="Total Water Consumption (Billion Cubic Meters)",
                    hue="Risk_Level", ax=ax)
    st.pyplot(fig)

with c5:
    fig, ax = plt.subplots(figsize=(4,3))
    sns.scatterplot(data=data,
                    x="Groundwater Depletion Rate (%)",
                    y="Total Water Consumption (Billion Cubic Meters)",
                    hue="Risk_Level", ax=ax)
    st.pyplot(fig)

with c6:
    fig, ax = plt.subplots(figsize=(4,3))
    sns.scatterplot(data=data,
                    x="Per Capita Water Use (Liters per Day)",
                    y="Groundwater Depletion Rate (%)",
                    hue="Risk_Level", ax=ax)
    st.pyplot(fig)

st.markdown("---")

# ==================================================
# ML MODELS
# ==================================================
features = [
    "Year",
    "Per Capita Water Use (Liters per Day)",
    "Agricultural Water Use (%)",
    "Industrial Water Use (%)",
    "Household Water Use (%)",
    "Rainfall Impact (Annual Precipitation in mm)",
    "Groundwater Depletion Rate (%)"
]

X = data[features]
y = data["Total Water Consumption (Billion Cubic Meters)"]

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)
rf_preds = rf.predict(X)

# ==================================================
# FEATURE IMPORTANCE
# ==================================================
st.subheader("🧠 Why the AI Predicts This")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(5,3))
sns.barplot(data=importance, x="Importance", y="Feature", ax=ax)
st.pyplot(fig)

top_feature = importance.iloc[0]["Feature"]
st.success(f"Key driver identified by AI: **{top_feature}**")

st.markdown("---")

# ==================================================
# CONFIDENCE BANDS
# ==================================================
residuals = y - rf_preds
std = residuals.std()

future_input = pd.DataFrame({
    "Year": future_years.flatten(),
    "Per Capita Water Use (Liters per Day)": data["Per Capita Water Use (Liters per Day)"].mean(),
    "Agricultural Water Use (%)": data["Agricultural Water Use (%)"].mean(),
    "Industrial Water Use (%)": data["Industrial Water Use (%)"].mean(),
    "Household Water Use (%)": data["Household Water Use (%)"].mean(),
    "Rainfall Impact (Annual Precipitation in mm)": data["Rainfall Impact (Annual Precipitation in mm)"].mean(),
    "Groundwater Depletion Rate (%)": data["Groundwater Depletion Rate (%)"].mean()
})

future_preds = rf.predict(future_input)
upper = future_preds + 1.96 * std
lower = future_preds - 1.96 * std

st.subheader("🔮 Forecast with Confidence Bands")

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(data["Year"], y, label="Historical")
ax.plot(future_years, future_preds, linestyle="--", marker="x", label="Forecast")
ax.fill_between(future_years.flatten(), lower, upper, alpha=0.3, label="95% CI")
ax.legend()
st.pyplot(fig)

st.markdown("---")

# ==================================================
# SCENARIO SIMULATOR
# ==================================================
st.subheader("🎛️ What-If Scenario Simulator")

rainfall_drop = st.slider("Rainfall Reduction (%)", 0, 50, 20)
demand_increase = st.slider("Demand Increase (%)", 0, 50, 15)

scenario = future_input.copy()
scenario["Rainfall Impact (Annual Precipitation in mm)"] *= (1 - rainfall_drop/100)
scenario["Per Capita Water Use (Liters per Day)"] *= (1 + demand_increase/100)

scenario_preds = rf.predict(scenario)

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(future_years, future_preds, label="Base Forecast")
ax.plot(future_years, scenario_preds, linestyle=":", marker="x", label="Scenario")
ax.legend()
st.pyplot(fig)

st.markdown("---")

# ==================================================
# COUNTRY COMPARISON
# ==================================================
st.subheader("🌍 Country Comparison")

cA, cB = st.columns(2)

with cA:
    country_a = st.selectbox("Country A", sorted(df["Country"].unique()), key="a")
    dA = df[df["Country"] == country_a]

with cB:
    country_b = st.selectbox("Country B", sorted(df["Country"].unique()), key="b")
    dB = df[df["Country"] == country_b]

st.metric(f"{country_a} Avg Demand",
          round(dA["Total Water Consumption (Billion Cubic Meters)"].mean(),2))
st.metric(f"{country_b} Avg Demand",
          round(dB["Total Water Consumption (Billion Cubic Meters)"].mean(),2))

st.markdown("---")

# ==================================================
# AI CHAT ADVISOR
# ==================================================
st.subheader("💬 Ask the Dashboard")

question = st.text_input("Ask a question:")

if question:
    q = question.lower()
    if "risk" in q:
        st.success(f"Dominant risk level: {data['Risk_Level'].mode()[0]}")
    elif "forecast" in q:
        st.success("AI predicts rising water demand under current trends.")
    elif "policy" in q:
        st.success("Focus on groundwater protection and efficient agriculture.")
    else:
        st.success("Water scarcity is predictable and manageable with early AI intervention.")

# ==================================================
# EXECUTIVE SUMMARY PDF
# ==================================================
st.subheader("📄 Executive Summary")

if st.button("Generate PDF"):
    file = f"Aquasense_Summary_{country}.pdf"
    c = canvas.Canvas(file, pagesize=A4)
    text = c.beginText(40, 800)
    text.textLine("Aquasense – Executive Summary")
    text.textLine(f"Country: {country}")
    text.textLine(f"Risk Level: {data['Risk_Level'].mode()[0]}")
    text.textLine(f"Top AI Driver: {top_feature}")
    text.textLine("Recommendation: Act early on demand and groundwater protection.")
    c.drawText(text)
    c.save()
    st.success("PDF generated successfully.")
