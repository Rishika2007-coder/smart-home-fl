import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import os
from model import EnergyModel, ActivityModel, load_client_data

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Home FL Dashboard",
    page_icon="🏠",
    layout="wide"
)

# ── Navigation ─────────────────────────────────────────────────────────
st.sidebar.title("🏠 Smart Home FL")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home Overview",
     "🤖 Global Model",
     "📊 Predictions",
     "🔒 Privacy Explained"]
)
st.sidebar.markdown("---")
selected_home = st.sidebar.selectbox(
    "Select Home",
    [1, 2, 3, 4, 5],
    format_func=lambda x: f"Home {x}"
)


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Home Overview
# ══════════════════════════════════════════════════════════════════════
if page == "🏠 Home Overview":
    st.title("🏠 Smart Home Overview")
    st.markdown("Sensor data collected from each smart home client.")
    st.divider()

    # Top metrics
    df = pd.read_csv(f"data/client_{selected_home}.csv")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Home",     f"Home {selected_home}")
    col2.metric("Total Samples",     len(df))
    col3.metric("Avg Energy (kWh)",  round(df["energy"].mean(), 2))
    col4.metric("Avg Temperature",   f"{round(df['temperature'].mean(), 1)}°C")
    st.divider()

    # Charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**Energy Consumption by Hour — Home {selected_home}**")
        hourly = df.groupby("hour")["energy"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(hourly["hour"], hourly["energy"],
               color="#185FA5", alpha=0.85)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg Energy (kWh)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown(f"**Activity Distribution — Home {selected_home}**")
        activity_counts = df["activity"].value_counts().sort_index()
        labels = {0: "Sleeping", 1: "Active", 2: "Away"}
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.pie(
            activity_counts.values,
            labels=[labels[i] for i in activity_counts.index],
            colors=["#185FA5", "#1D9E75", "#D85A30"],
            autopct="%1.1f%%",
            startangle=90
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Compare all homes
    st.markdown("**Energy Comparison Across All 5 Homes**")
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ["#185FA5", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517"]
    for home_id in range(1, 6):
        df_h = pd.read_csv(f"data/client_{home_id}.csv")
        hourly_h = df_h.groupby("hour")["energy"].mean()
        ax.plot(hourly_h.index, hourly_h.values,
                label=f"Home {home_id}",
                color=colors[home_id - 1],
                linewidth=1.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Energy (kWh)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Global Model (NEW PAGE)
# ══════════════════════════════════════════════════════════════════════
elif page == "🤖 Global Model":
    st.title("🤖 Global Model — Federated Learning Training")
    st.divider()

    # FL metrics charts
    if os.path.exists("logs/metrics.json"):
        with open("logs/metrics.json") as f:
            metrics = json.load(f)

        # Summary metrics
        st.subheader("Global Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total FL Rounds",      len(metrics["rounds"]))
        m2.metric("Total Homes (Clients)", "5")
        m3.metric(
            "Final Accuracy",
            f"{metrics['accuracy'][-1]:.2%}",
            delta=f"+{(metrics['accuracy'][-1] - metrics['accuracy'][0]):.2%} from Round 1"
        )
        m4.metric(
            "Final Energy MSE",
            f"{metrics['energy_mse'][-1]:.4f}",
            delta=f"{(metrics['energy_mse'][-1] - metrics['energy_mse'][0]):.4f} from Round 1",
            delta_color="inverse"
        )

        st.divider()

        # Round by round progress
        st.subheader("Training Progress — Round by Round")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Activity Detection Accuracy per Round**")
            st.caption("Higher is better — should increase each round")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(metrics["rounds"], metrics["accuracy"],
                    marker="o", color="#185FA5",
                    linewidth=2.5, markersize=7)
            ax.fill_between(metrics["rounds"], metrics["accuracy"],
                            alpha=0.1, color="#185FA5")
            ax.set_xlabel("FL Round")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(metrics["rounds"])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("**Energy Prediction MSE per Round**")
            st.caption("Lower is better — should decrease each round")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(metrics["rounds"], metrics["energy_mse"],
                    marker="s", color="#D85A30",
                    linewidth=2.5, markersize=7)
            ax.fill_between(metrics["rounds"], metrics["energy_mse"],
                            alpha=0.1, color="#D85A30")
            ax.set_xlabel("FL Round")
            ax.set_ylabel("Mean Squared Error")
            ax.set_xticks(metrics["rounds"])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()

        # Round by round table
        st.subheader("Detailed Round Metrics")
        rounds_df = pd.DataFrame({
            "Round":        metrics["rounds"],
            "Accuracy":     [f"{a:.2%}" for a in metrics["accuracy"]],
            "Energy MSE":   [f"{m:.4f}" for m in metrics["energy_mse"]],
            "Accuracy Trend": ["⬆ Improving" if i > 0 and
                               metrics["accuracy"][i] > metrics["accuracy"][i-1]
                               else ("— Same" if i > 0 and
                               metrics["accuracy"][i] == metrics["accuracy"][i-1]
                               else "—")
                               for i in range(len(metrics["rounds"]))]
        })
        st.dataframe(rounds_df, use_container_width=True, hide_index=True)

        st.divider()

        # How FedAvg works
        st.subheader("How FedAvg Aggregation Works")
        st.markdown("""
        After every round, the server combines weights from all 5 homes
        using a **weighted average** based on how many samples each home has:
```
        Global Weights = (Home1_weights × n1 + Home2_weights × n2 + ... + Home5_weights × n5)
                         ─────────────────────────────────────────────────────────────────────
                                           n1 + n2 + n3 + n4 + n5
```

        Where `n1, n2...` = number of training samples each home has.
        Homes with more data have slightly more influence on the global model.
        """)

    else:
        st.warning("⚠️ No training data found yet.")
        st.info(
            "You need to run the federated learning training first!\n\n"
            "**Step 1** — Open Terminal 1 and run:\n"
            "```\npython server.py\n```\n\n"
            "**Step 2** — Open 5 more terminals and run:\n"
            "```\npython client.py 1\n"
            "python client.py 2\n"
            "python client.py 3\n"
            "python client.py 4\n"
            "python client.py 5\n```\n\n"
            "Once all 10 rounds complete, refresh this page."
        )


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Predictions
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 Predictions":
    st.title("📊 Model Predictions")
    st.markdown(f"Showing predictions for **Home {selected_home}**")
    st.divider()

    energy_model   = EnergyModel()
    activity_model = ActivityModel()

    (X_train, y_e_train, y_a_train,
     X_test,  y_e_test,  y_a_test) = load_client_data(selected_home)

    energy_model.eval()
    activity_model.eval()

    with torch.no_grad():
        energy_preds   = energy_model(X_test).numpy().flatten()
        activity_preds = torch.argmax(
            activity_model(X_test), dim=1
        ).numpy()

    # Accuracy
    accuracy = float(
        (torch.tensor(activity_preds) == y_a_test).float().mean()
    )
    mse = float(
        np.mean((energy_preds - y_e_test.numpy().flatten()) ** 2)
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Activity Accuracy", f"{accuracy:.2%}")
    col2.metric("Energy MSE",        f"{mse:.4f}")
    col3.metric("Test Samples",      len(X_test))
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Energy: Actual vs Predicted**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = range(80)
        ax.plot(x, y_e_test.numpy().flatten()[:80],
                label="Actual",
                color="#185FA5", linewidth=1.8)
        ax.plot(x, energy_preds[:80],
                label="Predicted",
                color="#D85A30", linestyle="--", linewidth=1.8)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Energy (kWh)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Activity Detection Confusion Matrix**")
        from sklearn.metrics import confusion_matrix
        cm     = confusion_matrix(y_a_test.numpy(), activity_preds)
        labels = ["Sleeping", "Active", "Away"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — Privacy Explained
# ══════════════════════════════════════════════════════════════════════
elif page == "🔒 Privacy Explained":
    st.title("🔒 How Privacy is Preserved")
    st.divider()

    st.subheader("Traditional AI vs Federated Learning")
    col1, col2 = st.columns(2)

    with col1:
        st.error(
            "**❌ Traditional Centralized AI**\n\n"
            "- All raw sensor data sent to cloud\n"
            "- Your daily routines stored on servers\n"
            "- Risk of data leaks and breaches\n"
            "- Company owns your personal data\n"
            "- Single point of failure"
        )

    with col2:
        st.success(
            "**✅ Federated Learning (This Project)**\n\n"
            "- Raw data never leaves your home\n"
            "- Only model weights are shared\n"
            "- No personal data on servers\n"
            "- You keep full data ownership\n"
            "- Works even if server is compromised"
        )

    st.divider()
    st.subheader("What Data Stays Private?")

    p1, p2, p3, p4 = st.columns(4)
    p1.info("🌡️ **Temperature**\n\nYour climate preferences stay local")
    p2.info("🚶 **Motion**\n\nYour movement patterns stay local")
    p3.info("💡 **Light Usage**\n\nYour daily habits stay local")
    p4.info("⚡ **Energy**\n\nYour consumption data stays local")

    st.divider()
    st.subheader("What is Actually Shared with the Server?")
    st.markdown("""
    Only **model weights** — these are just floating point numbers
    that represent learned patterns. For example:
```
    Layer 1 weights: [0.234, -0.891, 0.445, 0.123, ...]
    Layer 2 weights: [-0.334, 0.221, 0.667, -0.445, ...]
```

    These numbers **cannot be reverse-engineered** back into
    your original sensor readings. Your privacy is mathematically guaranteed.
    """)

    st.divider()
    st.caption("Built with Flower (flwr) · PyTorch · Streamlit")