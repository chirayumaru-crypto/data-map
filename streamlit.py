import os
import re
import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------
st.set_page_config(page_title="AI Log + Transcript Analyzer", layout="wide")
st.title("🤖 AI Log + Transcript Analyzer")
st.caption("Upload your CSV log and VTT transcript to align, analyze, and visualize system-human interactions.")

# -----------------------------------------------------------
# Helper: remove illegal Excel characters
# -----------------------------------------------------------
def remove_illegal_chars(value):
    if isinstance(value, str):
        # Remove control chars not allowed in XML/Excel
        return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", value)
    return value

# -----------------------------------------------------------
# File Uploads
# -----------------------------------------------------------
uploaded_csv = st.file_uploader("📁 Upload CSV Log File", type=["csv"])
uploaded_vtt = st.file_uploader("🎙️ Upload VTT Transcript File", type=["vtt"])

if uploaded_csv and uploaded_vtt:

    # -----------------------------------------------------------
    # CSV Cleaning
    # -----------------------------------------------------------
    def clean_log_dataframe(file):
        df = pd.read_csv(file)
        df = df.drop_duplicates()
        df.columns = df.columns.str.strip()
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df["Direction"] = df["Direction"].fillna("UNKNOWN")
        df["Status"] = df["Status"].fillna("UNKNOWN")

        def clean_data(data):
            if pd.isna(data):
                return None
            try:
                decoded = bytes.fromhex(data.replace(" ", "")).decode('ascii', errors='ignore')
                return remove_illegal_chars(decoded)
            except Exception:
                return remove_illegal_chars(str(data))

        if "Data" in df.columns:
            df["Data_clean"] = df["Data"].apply(clean_data)
        return df

    logs = clean_log_dataframe(uploaded_csv)
    st.success(f"✅ CSV Loaded: {len(logs)} rows")

    # -----------------------------------------------------------
    # VTT Parsing
    # -----------------------------------------------------------
    def parse_vtt(file):
        content = file.read().decode("utf-8")
        pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\s+([\s\S]*?)(?=\n\d{2}:\d{2}:\d{2}\.\d{3}|\Z)",
            re.MULTILINE
        )
        matches = pattern.findall(content)
        vtt_df = pd.DataFrame(matches, columns=["start", "end", "text"])

        def to_seconds(t):
            h, m, s = t.split(":")
            return int(h)*3600 + int(m)*60 + float(s)

        vtt_df["start_sec"] = vtt_df["start"].apply(to_seconds)
        vtt_df["end_sec"] = vtt_df["end"].apply(to_seconds)
        vtt_df["text"] = vtt_df["text"].str.replace("\n", " ").str.strip()
        vtt_df["speaker"] = vtt_df["text"].apply(
            lambda x: x.split(":")[0].strip().title() if ":" in x else "Unknown"
        )
        vtt_df["content"] = vtt_df["text"].apply(
            lambda x: x.split(":", 1)[1].strip() if ":" in x else x
        )
        return vtt_df

    vtt_df = parse_vtt(uploaded_vtt)
    st.success(f"✅ VTT Parsed: {len(vtt_df)} captions | Speakers: {vtt_df['speaker'].nunique()}")

    # -----------------------------------------------------------
    # Time Alignment
    # -----------------------------------------------------------
    logs["Parsed_Time"] = pd.to_datetime(logs["Time"], errors="coerce", dayfirst=True)
    start_time = logs["Parsed_Time"].min()
    logs["sec_from_start"] = (logs["Parsed_Time"] - start_time).dt.total_seconds()

    def find_vtt_segment(time_sec):
        match = vtt_df[(vtt_df["start_sec"] <= time_sec) & (vtt_df["end_sec"] >= time_sec)]
        if not match.empty:
            return match.iloc[0]["text"]
        nearest = vtt_df.iloc[(vtt_df["start_sec"] - time_sec).abs().argsort()[:1]]
        if not nearest.empty and abs(nearest.iloc[0]["start_sec"] - time_sec) <= 2:
            return nearest.iloc[0]["text"]
        return None

    logs["Matched_Speech"] = logs["sec_from_start"].apply(find_vtt_segment)
    missing_count = logs["Matched_Speech"].isna().sum()
    st.info(f"🕒 Time-aligned matches: {len(logs)-missing_count} | Missing: {missing_count}")

    # -----------------------------------------------------------
    # Summary Metrics
    # -----------------------------------------------------------
    st.header("📈 Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Logs", len(logs))
    col2.metric("Total Speakers", vtt_df["speaker"].nunique())
    col3.metric("Speech Segments", len(vtt_df))

    col4, col5 = st.columns(2)
    col4.metric("Functions", logs["Function"].nunique())
    col5.metric("Directions", logs["Direction"].nunique())

    # -----------------------------------------------------------
    # Counts
    # -----------------------------------------------------------
    st.subheader("🧩 Function & Direction Analysis")
    func_counts = logs["Function"].value_counts().reset_index()
    func_counts.columns = ["Function", "Count"]
    st.bar_chart(func_counts.set_index("Function"))

    dir_counts = logs["Direction"].value_counts().reset_index()
    dir_counts.columns = ["Direction", "Count"]
    st.bar_chart(dir_counts.set_index("Direction"))

    # -----------------------------------------------------------
    # Keyword Extraction
    # -----------------------------------------------------------
    st.header("🗝️ Keyword Frequency from Transcript")
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(vtt_df["content"])
    keywords = pd.Series(X.toarray().sum(axis=0), index=vectorizer.get_feature_names_out()).sort_values(ascending=False)
    st.bar_chart(keywords.head(15))

    # -----------------------------------------------------------
    # 🎞️ TIMELINE VISUALIZATION (Plotly)
    # -----------------------------------------------------------
    st.header("🎞️ Timeline Playback Visualization")
    combined = pd.DataFrame({
        "Time (sec)": logs["sec_from_start"],
        "Log Event": 1
    }).groupby("Time (sec)").count().reset_index()

    speech_segments = vtt_df[["speaker", "start_sec", "end_sec"]].copy()

    fig = go.Figure()

    # Log activity line
    fig.add_trace(go.Scatter(
        x=combined["Time (sec)"],
        y=combined["Log Event"],
        mode='lines',
        name="Log Activity",
        line=dict(width=2, color='blue')
    ))

    # Add speech bars
    for _, row in speech_segments.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["start_sec"], row["end_sec"]],
            y=[0.5, 0.5],
            mode="lines",
            name=row["speaker"],
            line=dict(width=10),
            opacity=0.5
        ))

    fig.update_layout(
        title="System Activity and Speech Over Time",
        xaxis_title="Seconds from Start",
        yaxis_title="Event Frequency",
        showlegend=True,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------
    # 🧭 SPEAKER–FUNCTION MATRIX
    # -----------------------------------------------------------
    st.header("🧭 Speaker–Function Matrix")
    merged = logs.merge(vtt_df, left_on="Matched_Speech", right_on="text", how="left")
    matrix = pd.crosstab(merged["speaker"], merged["Function"])
    st.dataframe(matrix.style.background_gradient(cmap="Blues"))

    # -----------------------------------------------------------
    # Download Aligned Excel
    # -----------------------------------------------------------
    st.header("💾 Export Aligned Data")
    cleaned_logs = logs[["Time", "Function", "Direction", "Status", "Data_clean", "Matched_Speech"]].applymap(remove_illegal_chars)
    output = io.BytesIO()
    cleaned_logs.to_excel(output, index=False)
    st.download_button(
        label="⬇️ Download Clean Aligned Excel",
        data=output.getvalue(),
        file_name=f"aligned_output_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("⬆️ Please upload both a CSV and a VTT file to begin analysis.")

