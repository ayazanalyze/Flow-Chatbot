import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Groundwater Insights Dashboard", layout="wide")

# Load your cleaned CSV (adjust path if needed)
df = pd.read_csv("cleaned_groundwater_data.csv")

st.title("📊 Groundwater Insights Dashboard")
st.caption("Deep dive into India's groundwater data. Explore risk, trends, and actionable insights.")

# Risk category pie chart
def classify_row(stage):
    if stage > 90:
        return "Over-Exploited"
    elif stage > 70:
        return "Critical"
    else:
        return "Safe"

# CREATE the Category column FIRST
df['Category'] = df['ExtractionStage_Percent'].apply(classify_row)

# Now you can use the Category column
category_counts = df['Category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']
fig_pie = px.pie(
    category_counts,
    names='Category',
    values='Count',
    color_discrete_map={"Over-Exploited":"#ff4f4f", "Critical":"#f3ba2f", "Safe":"#03fc7f"},
    title="Groundwater Risk Categories"
)

# Top 10 districts bar chart
top_districts = df.groupby('District')['ExtractionStage_Percent'].mean().sort_values(ascending=False).head(10).reset_index()
fig_bar = px.bar(top_districts, x='District', y='ExtractionStage_Percent', color='ExtractionStage_Percent',
                 color_continuous_scale='Reds', title="Top 10 Districts by Groundwater Extraction (%)")
fig_bar.update_xaxes(tickangle=45)

# Statewise average extraction heatmap
states_avg = df.groupby('State')['ExtractionStage_Percent'].mean().reset_index()
fig_state = px.bar(states_avg, x="State", y="ExtractionStage_Percent", color="ExtractionStage_Percent",
                   color_continuous_scale='Plasma', title="Statewise Average Groundwater Extraction (%)")
fig_state.update_xaxes(tickangle=45)

# Water availability vs extraction scatter
fig_scatter = px.scatter(
    df,
    x="GroundWaterAvailability_ham",
    y="ExtractionStage_Percent",
    color="Category",
    hover_data=["District", "State"],
    title="Groundwater Availability vs Extraction (%) by Block"
)

# Interactive table comparison
st.subheader("📋 District Comparison Table")
st.dataframe(df.groupby(["State","District"]).agg({
    "GroundWaterAvailability_ham": "mean",
    "ExtractionStage_Percent": "mean",
    "Rainfall_mm": "mean"
}).reset_index().round(1))

# Display dashboard charts
st.subheader("🔎 Category Risk Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("🏆 Top 10 Risky Districts")
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("🌍 Statewise Groundwater Stress")
st.plotly_chart(fig_state, use_container_width=True)

st.subheader("💧 Water Availability vs Extraction (Scatter)")
st.plotly_chart(fig_scatter, use_container_width=True)

st.caption("All data powered by the INGRES portal | SIH 2025 hackathon")
