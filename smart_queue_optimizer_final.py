import streamlit as st
import pandas as pd
import time
from sklearn.linear_model import LinearRegression

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Smart Queue Optimizer",
    page_icon="🎟️",
    layout="centered"
)

# ------------------ HEADER ------------------
st.markdown("<h1 style='text-align:center;'>🎟️ Smart Queue Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AI-Based Real-Time Queue Management System</h4>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ SAMPLE DATASET ------------------
data = {
    "customers": [10, 20, 30, 40, 50, 60, 70],
    "counters": [1, 2, 2, 3, 3, 4, 4],
    "service_time": [5, 5, 6, 6, 7, 7, 8],
    "waiting_time": [50, 45, 60, 55, 70, 65, 80]
}

df = pd.DataFrame(data)

# ------------------ ML MODEL ------------------
X = df[["customers", "counters", "service_time"]]
y = df["waiting_time"]

model = LinearRegression()
model.fit(X, y)

# ------------------ USER INPUT ------------------
st.subheader("🔢 Enter Queue Details")

total_tokens = st.number_input("Total Tokens Issued", min_value=1, value=20)
counters = st.number_input("Active Counters", min_value=1, value=2)
avg_service_time = st.number_input("Average Service Time (minutes)", min_value=1, value=5)

your_token = st.number_input("🎫 Your Token Number", min_value=1, max_value=total_tokens, value=5)

# ------------------ REAL-TIME SIMULATION ------------------
st.subheader("⏱️ Live Queue Status")

current_token_placeholder = st.empty()
status_placeholder = st.empty()
notify_placeholder = st.empty()

current_token = 1

for token in range(1, your_token + 1):
    remaining = total_tokens - token

    predicted_wait = model.predict([[remaining, counters, avg_service_time]])[0]

    current_token_placeholder.info(f"🔔 Current Token Being Served: **{token}**")
    status_placeholder.write(
        f"📌 Remaining in Queue: **{remaining}** | ⏳ Predicted Wait: **{int(predicted_wait)} minutes**"
    )

    # ------------------ NOTIFICATION FEATURE ------------------
    if token == your_token - 1:
        notify_placeholder.warning("⚠️ **Get Ready! Only ONE token left before your turn!**")

    if token == your_token:
        notify_placeholder.success("✅ **It's your turn now! Please proceed to the counter.**")
        st.balloons()
        break

    time.sleep(2)

# ------------------ FINAL WAIT TIME DISPLAY ------------------
st.markdown("---")
st.subheader("📊 Final Prediction")

final_wait = model.predict([[your_token, counters, avg_service_time]])[0]

if final_wait < 30:
    st.success(f"🎉 Estimated Waiting Time: **{int(final_wait)} minutes**")
else:
    st.error(f"⌛ Estimated Waiting Time: **{int(final_wait)} minutes**")

# ------------------ DOWNLOAD DATASET ------------------
st.markdown("---")
st.subheader("⬇️ Download Dataset")

st.download_button(
    label="Download Queue Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="queue_dataset.csv",
    mime="text/csv"
)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Final Year Project | Smart Queue Optimizer | AI + Data Analytics")