import pandas as pd
import matplotlib.pyplot as plt

PHASE_ORDER = ["No Memory", "Base", "WB1", "WB2", "WB3", "WB4"]

def read_metrics_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1)
    df["Memory Phase"] = df["Memory Phase"].astype(str).str.strip()
    df["Metrics (avg)"] = df["Metrics (avg)"].astype(str).str.strip().str.upper()

    df["Memory Phase"] = pd.Categorical(
        df["Memory Phase"], categories=PHASE_ORDER, ordered=True
    )

    for col in ["Base Model", "FT", "MoE-FT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_f1_series(df: pd.DataFrame, col: str):
    d = df[df["Metrics (avg)"] == "F1"].copy()
    d = d.groupby("Memory Phase", as_index=False)[col].mean()
    d = d.set_index("Memory Phase").reindex(PHASE_ORDER)

    x = PHASE_ORDER
    y = d[col].tolist()
    return x, y


def plot_single_model(E_csv, EV_csv, EVS_csv, model_col, title, out_png, dpi=1200):

    dfE = read_metrics_csv(E_csv)
    dfEV = read_metrics_csv(EV_csv)
    dfEVS = read_metrics_csv(EVS_csv)

    xE, yE = get_f1_series(dfE, model_col)
    xEV, yEV = get_f1_series(dfEV, model_col)
    xEVS, yEVS = get_f1_series(dfEVS, model_col)

    plt.figure(figsize=(6.5, 4.8))  # slightly larger for clarity

    plt.plot(xE, yE, marker="o", linewidth=3.0, label="E")
    plt.plot(xEV, yEV, marker="s", linewidth=3.0, label="E+V")
    plt.plot(xEVS, yEVS, marker="^", linewidth=3.2, label="E+V+S")

    plt.title(title, fontsize=14)
    plt.xlabel("Memory Phase", fontsize=12)
    plt.ylabel("F1 (avg)", fontsize=12)
    plt.xticks(rotation=25)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_png} (dpi={dpi})")


# ---------------- RUN ----------------

plot_single_model(
    E_csv="E.csv",
    EV_csv="EV.csv",
    EVS_csv="SEV.csv",
    model_col="Base Model",
    title="(a) Base Model — Writeback Progression (F1)",
    out_png="subplot1_base_model_F1.png",
    dpi=1200
)

plot_single_model(
    E_csv="E.csv",
    EV_csv="EV.csv",
    EVS_csv="SEV.csv",
    model_col="FT",
    title="(b) Fine-Tuned Model — Writeback Progression (F1)",
    out_png="subplot2_FT_model_F1.png",
    dpi=1200
)

plot_single_model(
    E_csv="E.csv",
    EV_csv="EV.csv",
    EVS_csv="SEV.csv",
    model_col="MoE-FT",
    title="(c) MoE Fine-Tuned Model — Writeback Progression (F1)",
    out_png="subplot3_MoE_model_F1.png",
    dpi=1200
)
