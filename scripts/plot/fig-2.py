import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Figure 2: Memory Combination Impact (Grouped Bars)
# CSV format expected (your table-style CSV):
#   Memory Combinations, Metrics (avg), Base, FT, MoE-FT, (optional delta cols...)
# rows include both F1 and EM per memory combination
# ----------------------------

# Optional: enforce a consistent display order (edit if you want different order)
COMBO_ORDER = ["None", "S", "E", "V", "S+E", "S+V", "E+V", "S+E+V"]

def read_table_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # normalize column names (handles accidental extra spaces)
    df.columns = [c.strip() for c in df.columns]

    # normalize strings
    df["Memory Combinations"] = df["Memory Combinations"].astype(str).str.strip()
    df["Metrics (avg)"] = df["Metrics (avg)"].astype(str).str.strip().str.upper()

    # numeric columns (robust)
    for col in ["Base", "FT", "MoE-FT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # enforce order if possible
    df["Memory Combinations"] = pd.Categorical(
        df["Memory Combinations"], categories=COMBO_ORDER, ordered=True
    )

    return df


def plot_figure2_grouped_bars(
    csv_path: str,
    metric: str = "F1",   # "F1" or "EM"
    subplot: str = "(a)",
    out_png: str = "Figure2_WB4_groupedbars_F1.png",
    dpi: int = 1200,
    show_delta: bool = False,   # if True: add ΔFT and Δ(MoE-FT) annotations on bars
):
    df = read_table_csv(csv_path)

    # filter to metric
    metric = metric.upper()
    d = df[df["Metrics (avg)"] == metric].sort_values("Memory Combinations")

    # drop combos not in COMBO_ORDER (or missing)
    d = d.dropna(subset=["Memory Combinations"])

    combos = d["Memory Combinations"].astype(str).tolist()
    y_base = d["Base"].tolist()
    y_ft = d["FT"].tolist()
    y_moe = d["MoE-FT"].tolist()

    x = list(range(len(combos)))
    w = 0.26

    plt.figure(figsize=(10.2, 4.9))

    bars_base = plt.bar([i - w for i in x], y_base, width=w, label="Base")
    bars_ft   = plt.bar([i for i in x],     y_ft,   width=w, label="FT")
    bars_moe  = plt.bar([i + w for i in x], y_moe,  width=w, label="MoE-FT")

    plt.xticks(x, combos)
    plt.xlabel("Memory Combination")
    plt.ylabel(f"{metric} (avg)")
    plt.title(f"{subplot} Memory Combination impact at WB4 ({metric})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=False)

    # Optional: annotate deltas (absolute gains over Base)
    if show_delta:
        for i in range(len(combos)):
            b = y_base[i]
            f = y_ft[i]
            m = y_moe[i]
            if pd.notna(b) and pd.notna(f):
                plt.text(i, f, f"+{(f-b):.2f}", ha="center", va="bottom", fontsize=9)
            if pd.notna(b) and pd.notna(m):
                plt.text(i + w, m, f"+{(m-b):.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png} (dpi={dpi})")


if __name__ == "__main__":
    # Example usage:
    # 1) F1 figure
    plot_figure2_grouped_bars(
        csv_path="figure2.csv",   # <-- change to your filename
        metric="F1",
        subplot="(a)",
        out_png="Figure2_WB4_F1_groupedbars.png",
        dpi=1200,
        show_delta=False
    )

    # 2) EM figure (optional)
    plot_figure2_grouped_bars(
        csv_path="figure2.csv",   # <-- change to your filename
        metric="EM",
        subplot="(b)",
        out_png="Figure2_WB4_EM_groupedbars.png",
        dpi=1200,
        show_delta=False
    )
