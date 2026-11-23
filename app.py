
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="HIKARI NIDS",
    page_icon="🛡️",
    layout="wide"
)

MODEL_DIR = "models"

BINARY_MODEL_PATH = os.path.join(MODEL_DIR, "binary_model.pkl")
MULTI_MODEL_PATH = os.path.join(MODEL_DIR, "multiclass_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# =====================================================
# 2. FEATURE CONFIGURATION
# =====================================================

FEATURE_LABELS = {
    "originp": "Source Port",
    "responp": "Destination Port",
    "flow_duration": "Flow Duration",
    "originh": "Source IP Address",
    "responh": "Destination IP Address"
}

COMMON_PORTS = {
    "Custom Port": None,
    "FTP (20)": 20,
    "FTP (21)": 21,
    "SSH (22)": 22,
    "Telnet (23)": 23,
    "SMTP (25)": 25,
    "DNS (53)": 53,
    "HTTP (80)": 80,
    "POP3 (110)": 110,
    "NTP (123)": 123,
    "IMAP (143)": 143,
    "HTTPS (443)": 443,
    "SMB (445)": 445,
    "MySQL (3306)": 3306,
    "RDP (3389)": 3389,
    "PostgreSQL (5432)": 5432,
    "HTTP Alternative (8080)": 8080
}


def friendly_name(feature):
    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").title()
    )


# =====================================================
# 3. LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    paths = [
        BINARY_MODEL_PATH,
        MULTI_MODEL_PATH,
        METADATA_PATH
    ]

    if not all(os.path.exists(p) for p in paths):
        return None, None, None

    binary = joblib.load(paths[0])
    multiclass = joblib.load(paths[1])

    with open(paths[2], "r") as file:
        metadata = json.load(file)

    return binary, multiclass, metadata


try:
    binary_model, multiclass_model, metadata = load_models()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


if binary_model is None or multiclass_model is None:
    st.error("Models not found. Run train_model.py first.")
    st.stop()


# =====================================================
# 4. VALIDATE METADATA
# =====================================================

required = [
    "feature_columns",
    "demo",
    "multiclass_metrics",
    "multiclass_comparison"
]

if any(key not in metadata for key in required):
    st.error("Model metadata is incomplete. Retrain the models.")
    st.stop()

demo = metadata["demo"]

if not all(
    key in demo for key in ["defaults", "numeric_features", "options"]
):
    st.error("Demo metadata is incomplete.")
    st.stop()


# =====================================================
# 5. INPUT PREPARATION
# =====================================================

def prepare_input(df):

    prepared = df.copy()

    for col in metadata["feature_columns"]:
        if col not in prepared.columns:
            prepared[col] = demo["defaults"].get(col, np.nan)

    prepared = prepared[metadata["feature_columns"]].copy()

    for col in demo["numeric_features"]:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(
                prepared[col], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)

    return prepared


# =====================================================
# 6. PREDICTION
# =====================================================

def predict_traffic(df):

    data = prepare_input(df)

    bp = binary_model.predict(data)[0]
    mp = multiclass_model.predict(data)[0]

    bprob = binary_model.predict_proba(data)[0]
    mprob = multiclass_model.predict_proba(data)[0]

    return {
        "binary_prediction": str(bp),
        "multiclass_prediction": str(mp),

        "binary_probabilities": dict(zip(
            map(str, binary_model.classes_),
            map(float, bprob)
        )),

        "multiclass_probabilities": dict(zip(
            map(str, multiclass_model.classes_),
            map(float, mprob)
        ))
    }


# =====================================================
# 7. CONSISTENCY CHECK
# =====================================================

def check_consistency(binary, multiclass):

    benign = ["benign", "normal", "background"]

    b = str(binary).strip().lower() in benign
    m = str(multiclass).strip().lower() in benign

    return b == m


# =====================================================
# 8. DISPLAY PREDICTION
# =====================================================

def display_prediction(result):

    bp = result["binary_prediction"]
    mp = result["multiclass_prediction"]

    if check_consistency(bp, mp):
        st.success("Binary and Multiclass predictions are consistent.")
    else:
        st.warning("Prediction inconsistency detected. Investigate this traffic.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Binary Classification")

        if bp.strip().lower() in ["benign", "normal"]:
            st.success("🟢 BENIGN TRAFFIC")
        else:
            st.error("🚨 ATTACK DETECTED")

        st.metric(
            "Binary Confidence",
            f"{max(result['binary_probabilities'].values()) * 100:.2f}%"
        )

    with col2:
        st.subheader("Traffic Category")

        if mp.strip().lower() in ["benign", "normal", "background"]:
            st.success(f"🟢 {mp.upper()}")
        else:
            st.error(f"🚨 {mp.upper()}")

        st.metric(
            "Multiclass Confidence",
            f"{max(result['multiclass_probabilities'].values()) * 100:.2f}%"
        )

    st.divider()
    st.subheader("Traffic Category Probability Distribution")

    probabilities = pd.Series(
        result["multiclass_probabilities"]
    ) * 100

    st.bar_chart(probabilities)

    st.caption(
        "Model probabilities are not necessarily calibrated confidence estimates."
    )


# =====================================================
# 9. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🛡️ HIKARI NIDS")
st.sidebar.caption("Machine Learning-Based Network Intrusion Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Traffic Simulator",
        "Real Dataset Prediction",
        "Batch CSV Prediction"
    ]
)

st.sidebar.divider()
st.sidebar.info("Based on the HIKARI 2021 dataset.")

# =====================================================
# 10. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🛡️ HIKARI Network Intrusion Detection")

    st.markdown("### Machine Learning-Based Network Traffic Analysis")

    st.write(
        "Detect potentially malicious network traffic "
        "and identify traffic categories using machine learning."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Binary Classes", len(binary_model.classes_))
    c2.metric("Traffic Categories", len(multiclass_model.classes_))
    c3.metric("ML Algorithms", 4)

    st.divider()
    st.subheader("Available Traffic Categories")

    categories = multiclass_model.classes_
    cols = st.columns(min(3, len(categories)))

    for i, category in enumerate(categories):
        cols[i % len(cols)].info(f"🔍 {category}")

    st.divider()
    st.subheader("Model Performance")

    metrics = metadata["multiclass_metrics"]
    cols = st.columns(4)

    for col, name in zip(
        cols,
        ["Accuracy", "Precision", "Recall", "F1-score"]
    ):
        value = metrics.get(name)

        col.metric(
            name,
            f"{value * 100:.2f}%" if value is not None else "N/A"
        )

    st.caption("Metrics from the independent test dataset.")

    st.subheader("Model Comparison")

    st.dataframe(
        pd.DataFrame(metadata["multiclass_comparison"]),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 11. TRAFFIC SIMULATOR
# =====================================================

elif page == "Traffic Simulator":

    st.title("🔬 Network Traffic Prediction Simulator")

    st.write(
        "Enter the five network traffic features below "
        "to predict whether the traffic is benign or malicious."
    )

    st.info(
        "Only five features are displayed. Remaining model "
        "features use training metadata defaults."
    )

    traffic_input = {}

    col1, col2 = st.columns(2)

    # SOURCE PORT
    with col1:

        st.subheader("Source Port")

        source_option = st.selectbox(
            "Select Source Port",
            list(COMMON_PORTS.keys()),
            key="source_port"
        )

        if source_option == "Custom Port":
            source_port = st.number_input(
                "Enter Source Port",
                min_value=0,
                max_value=65535,
                value=51524,
                step=1
            )
        else:
            source_port = COMMON_PORTS[source_option]

        traffic_input["originp"] = int(source_port)

    # DESTINATION PORT
    with col2:

        st.subheader("Destination Port")

        destination_option = st.selectbox(
            "Select Destination Port",
            list(COMMON_PORTS.keys()),
            index=list(COMMON_PORTS.keys()).index("HTTPS (443)"),
            key="destination_port"
        )

        if destination_option == "Custom Port":
            destination_port = st.number_input(
                "Enter Destination Port",
                min_value=0,
                max_value=65535,
                value=443,
                step=1
            )
        else:
            destination_port = COMMON_PORTS[destination_option]

        traffic_input["responp"] = int(destination_port)

    # FLOW DURATION
    with col1:

        st.subheader("Flow Duration")

        flow_duration = st.number_input(
            "Enter Flow Duration",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="Use the same unit as the training dataset."
        )

        traffic_input["flow_duration"] = flow_duration

    # SOURCE IP
    with col2:

        st.subheader("Source IP Address")

        source_ip = st.text_input(
            "Enter Source IP",
            value="192.168.1.10"
        )

        traffic_input["originh"] = source_ip

    # DESTINATION IP
    with col1:

        st.subheader("Destination IP Address")

        destination_ip = st.text_input(
            "Enter Destination IP",
            value="192.168.1.20"
        )

        traffic_input["responh"] = destination_ip

    st.divider()

    # SELECTED TRAFFIC PROFILE
    st.subheader("Selected Traffic Profile")

    preview = pd.DataFrame([
        {
            "Feature": friendly_name(feature),
            "Selected Value": value
        }
        for feature, value in traffic_input.items()
    ])

    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True
    )

    # PREDICTION
    if st.button(
        "🚀 Predict Traffic",
        type="primary",
        use_container_width=True
    ):

        try:

            input_df = pd.DataFrame([traffic_input])

            result = predict_traffic(input_df)

            st.session_state["last_prediction"] = result

            st.divider()

            display_prediction(result)

            st.info(
                "This is a simulated feature combination. "
                "Validate important results using real traffic records."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# =====================================================
# 12. REAL DATASET PREDICTION
# =====================================================


# =====================================================
# REAL DATASET PREDICTION - MEMORY EFFICIENT
# =====================================================

elif page == "Real Dataset Prediction":

    st.title("🔍 Real Network Traffic Prediction")

    st.write("Upload a HIKARI dataset and search using its UID.")

    uploaded_file = st.file_uploader(
        "Upload HIKARI CSV",
        type=["csv"],
        key="uid_upload"
    )

    if uploaded_file:

        uid = st.text_input("Enter Network Flow UID")

        if st.button("Search and Predict", type="primary"):

            try:
                uploaded_file.seek(0)

                found_row = None
                chunk_size = 5000

                progress = st.progress(0)
                status = st.empty()

                # Read CSV in smaller chunks
                for chunk_number, chunk in enumerate(
                    pd.read_csv(
                        uploaded_file,
                        chunksize=chunk_size,
                        low_memory=True
                    )
                ):

                    chunk.columns = chunk.columns.str.strip()

                    uid_col = next(
                        (
                            col for col in chunk.columns
                            if col.lower() == "uid"
                        ),
                        None
                    )

                    if uid_col is None:
                        st.error("UID column not found in dataset.")
                        break

                    matches = chunk[
                        chunk[uid_col].astype(str) == uid.strip()
                    ]

                    if not matches.empty:
                        found_row = matches.iloc[[0]]
                        break

                    status.info(
                        f"Searching chunk {chunk_number + 1}..."
                    )

                progress.progress(100)

                if found_row is None:

                    st.error("UID not found in the dataset.")

                else:

                    st.success("Network flow found!")

                    input_df = found_row.drop(
                        columns=[
                            col for col in [
                                uid_col,
                                "Label",
                                "traffic_category"
                            ]
                            if col in found_row.columns
                        ],
                        errors="ignore"
                    )

                    # Use the five required features
                    required_features = [
                        "originp",
                        "responp",
                        "flow_duration",
                        "originh",
                        "responh"
                    ]

                    st.subheader("Selected Network Traffic Details")

                    available_features = [
                        f for f in required_features
                        if f in found_row.columns
                    ]

                    st.dataframe(
                        found_row[available_features],
                        use_container_width=True,
                        hide_index=True
                    )

                    result = predict_traffic(input_df)

                    display_prediction(result)

            except Exception as error:

                st.error(
                    f"Dataset processing failed: {error}"
                )


# =====================================================
# 13. BATCH CSV PREDICTION
# =====================================================


# =====================================================
# BATCH CSV PREDICTION - MEMORY EFFICIENT
# =====================================================

elif page == "Batch CSV Prediction":

    st.title("📂 Batch Network Traffic Prediction")

    st.write(
        "Upload a large HIKARI CSV and predict network "
        "traffic in smaller batches to reduce memory usage."
    )

    uploaded_file = st.file_uploader(
        "Upload Traffic CSV",
        type=["csv"],
        key="batch_upload"
    )

    if uploaded_file:

        if st.button("🚀 Predict All Traffic", type="primary"):

            try:

                uploaded_file.seek(0)

                chunk_size = 2000

                all_results = []
                total_records = 0

                benign_count = 0
                attack_count = 0
                inconsistent_count = 0

                progress = st.progress(0)
                status = st.empty()

                # Read CSV in smaller chunks
                reader = pd.read_csv(
                    uploaded_file,
                    chunksize=chunk_size,
                    low_memory=True
                )

                # First pass: count records for progress
                # is avoided to prevent reading the CSV twice.

                for chunk_number, chunk in enumerate(reader):

                    chunk.columns = chunk.columns.str.strip()

                    # Prepare model input
                    prepared = prepare_input(chunk)

                    # Predictions
                    bp = binary_model.predict(prepared)
                    mp = multiclass_model.predict(prepared)

                    probabilities = binary_model.predict_proba(
                        prepared
                    )

                    # Attack probability
                    attack_index = next(
                        (
                            i for i, label in enumerate(
                                binary_model.classes_
                            )
                            if str(label).strip().lower() == "attack"
                        ),
                        None
                    )

                    if attack_index is not None:
                        attack_probability = (
                            probabilities[:, attack_index] * 100
                        )
                    else:
                        attack_probability = np.full(
                            len(chunk), np.nan
                        )

                    # Consistency
                    consistency = [
                        "Consistent" if check_consistency(b, m)
                        else "Inconsistent"
                        for b, m in zip(bp, mp)
                    ]

                    # Results for current chunk
                    results = chunk.copy()

                    results["Binary_Prediction"] = bp
                    results["Traffic_Category"] = mp

                    results["Attack_Probability_Percent"] = (
                        attack_probability
                    )

                    results["Prediction_Consistency"] = consistency

                    # Store results
                    all_results.append(results)

                    # Update counts
                    total_records += len(chunk)

                    benign_count += sum(
                        str(x).strip().lower()
                        in ["benign", "normal"]
                        for x in bp
                    )

                    attack_count += len(chunk) - sum(
                        str(x).strip().lower()
                        in ["benign", "normal"]
                        for x in bp
                    )

                    inconsistent_count += consistency.count(
                        "Inconsistent"
                    )

                    status.info(
                        f"Processed {total_records:,} records..."
                    )

                    # Approximate progress indicator
                    progress.progress(
                        min(95, (chunk_number + 1) * 2)
                    )

                if not all_results:

                    st.warning("The uploaded CSV contains no records.")

                else:

                    # Combine predictions
                    final_results = pd.concat(
                        all_results,
                        ignore_index=True
                    )

                    st.session_state["batch_results"] = final_results

                    progress.progress(100)

                    status.success("Batch prediction completed!")

                    # Summary
                    st.subheader("Prediction Summary")

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric("Total Records", f"{total_records:,}")
                    c2.metric("Benign Records", f"{benign_count:,}")
                    c3.metric("Attack Records", f"{attack_count:,}")

                    c4.metric(
                        "Inconsistent Predictions",
                        f"{inconsistent_count:,}"
                    )

                    if inconsistent_count:
                        st.warning(
                            f"{inconsistent_count} records have "
                            "contradictory predictions."
                        )
                    else:
                        st.success("All predictions are consistent.")

                    # Traffic distribution
                    st.subheader("Traffic Distribution")

                    st.bar_chart(
                        final_results["Traffic_Category"].value_counts()
                    )

                    # Results preview
                    st.subheader("Prediction Results")

                    st.dataframe(
                        final_results.head(100),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Download
                    csv = final_results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Prediction Results",
                        data=csv,
                        file_name="hikari_predictions.csv",
                        mime="text/csv"
                    )

            except MemoryError:

                st.error(
                    "Not enough RAM. Reduce chunk_size from "
                    "2000 to 500 or 1000."
                )

            except Exception as error:

                st.error(f"Prediction failed: {error}")


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.caption(
    "HIKARI NIDS | Machine Learning-Based Network "
    "Intrusion Detection System | Academic Project"
)
