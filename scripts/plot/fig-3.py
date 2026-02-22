import pandas as pd
import matplotlib.pyplot as plt

COMBO_ORDER = ["None", "S", "E", "V", "S+E", "S+V", "E+V", "S+E+V"]

def _norm(s: str) -> str:
    return "".join(str(s).strip().lower().split())

def find_header_row(csv_path: str, max_scan_lines: int = 40) -> int:
    """
    Scan first N lines and return the 0-based index of the line
    that looks like the real CSV header.
    """
    required = ["memory", "metric", "ft", "moe"]  # "base" sometimes "base model"
    base_tokens = ["base", "basemodel"]

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [next(f, "") for _ in range(max_scan_lines)]

    best_idx = None
    best_score = -1

    for i, line in enumerate(lines):
        s = _norm(line)
        if not s:
            continue

        # score how many key tokens appear
        score = 0
        for t in required:
            if t in s:
                score += 1
        if any(bt in s for bt in base_tokens):
            score += 1

        # must include at least: memory + metric + ft + moe
        if "memory" in s and "metric" in s and "ft" in s and "moe" in s:
            # likely header
            if score > best_score:
                best_score = score
                best_idx = i

    if best_idx is None:
        # As a fallback, print lines to help debugging
        print("\n[DEBUG] Could not auto-detect header row. First lines:")
        for j, l in enumerate(lines):
            print(f"{j:02d}: {l.rstrip()}")
        raise RuntimeError(
            "Header row not found. Your CSV may not be a real comma-separated file "
            "(could be tab-separated or exported with weird delimiters)."
        )

    return best_idx

def read_custom_table_csv(csv_path: str, verbose: bool = True) -> pd.DataFrame:
    header_row = find_header_row(csv_path)

    if verbose:
        print(f"[DEBUG] Detected header row at line index: {header_row}")

    # Read using the detected header line
    df = pd.read_csv(csv_path, header=header_row)

    # Clean columns, drop unnamed
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", na=False)]

    if verbose:
        print("\n[DEBUG] Columns read:")
        for c in df.columns:
            print(" -", repr(c))

    # Normalize column name variations
    # Find likely columns by fuzzy contains
    cols_norm = {_norm(c): c for c in df.columns}

    def pick(cands):
        for cand in cands:
            cn = _norm(cand)
            # exact
            if cn in cols_norm:
                return cols_norm[cn]
            # contains
            for k, orig in cols_norm.items():
                if cn in k or k in cn:
                    return orig
        return None

    combo_col  = pick(["Memory Combinations", "Memory Combination", "Memory"])
    metric_col = pick(["Metrics (avg)", "Metrics", "Metric"])
    base_col   = pick(["Base Model", "Base"])
    ft_col     = pick(["FT"])
    moe_col    = pick(["MoE-FT", "MOE-FT", "MoE"])

    missing = []
    if combo_col is None:  missing.append("Memory Combinations column")
    if metric_col is None: missing.append("Metrics column")
    if base_col is None:   missing.append("Base column")
    if ft_col is None:     missing.append("FT column")
    if moe_col is None:    missing.append("MoE-FT column")
    if missing:
        raise KeyError(
            "Could not detect required columns: " + ", ".join(missing) +
            f"\n[Hint] Detected columns: {df.columns.tolist()}"
        )

    df = df.rename(columns={
        combo_col: "Memory Combinations",
        metric_col: "Metrics (avg)",
        base_col: "Base",
        ft_col: "FT",
        moe_col: "MoE-FT",
    })

    # Drop stray "Model" row if present
    df["Memory Combinations"] = df["Memory Combinations"].astype(str)
    df = df[df["Memory Combinations"].str.strip().str.lower() != "model"]

    # Forward fill combos for EM rows
    df["Memory Combinations"] = df["Memory Combinations"].replace({"nan": pd.NA, "": pd.NA}).ffill()

    # Clean strings
    df["Memory Combinations"] = df["Memory Combinations"].astype(str).str.strip()
    df["Metrics (avg)"] = df["Metrics (avg)"].astype(str).str.strip().str.upper()

    # Numeric
    for c in ["Base", "FT", "MoE-FT"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only rows that look like real entries
    df = df.dropna(subset=["Metrics (avg)"], how="any")

    # Order combos
    df["Memory Combinations"] = pd.Categorical(
        df["Memory Combinations"], categories=COMBO_ORDER, ordered=True
    )

    return df

def plot_figure2_grouped_bars(csv_path: str, metric="F1",
                              out_png="Figure2_WB4_groupedbars_F1.png", dpi=1200):
    df = read_custom_table_csv(csv_path, verbose=True)
    metric = metric.upper()

    d = df[df["Metrics (avg)"] == metric].sort_values("Memory Combinations")
    d = d.dropna(subset=["Memory Combinations"])

    combos = d["Memory Combinations"].astype(str).tolist()
    y_base = d["Base"].tolist()
    y_ft   = d["FT"].tolist()
    y_moe  = d["MoE-FT"].tolist()

    x = list(range(len(combos)))
    w = 0.25

    plt.figure(figsize=(9.6, 4.8))
    plt.bar([i - w for i in x], y_base, width=w, label="Base")
    plt.bar([i for i in x],     y_ft,   width=w, label="FT")
    plt.bar([i + w for i in x], y_moe,  width=w, label="MoE-FT")

    plt.xticks(x, combos)
    plt.xlabel("Memory combination")
    plt.ylabel(f"{metric} (avg)")
    plt.title(f"Figure 2: Memory combination impact at WB4 ({metric})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png} (dpi={dpi})")

def plot_figure3_moe_gain(csv_path: str, metric="F1", subplot="(a)",
                          out_png="Figure3_WB4_MoE_minus_FT_gain_F1.png",
                          dpi=1200, annotate=True):
    df = read_custom_table_csv(csv_path, verbose=False)
    metric = metric.upper()

    d = df[df["Metrics (avg)"] == metric].sort_values("Memory Combinations")
    d = d.dropna(subset=["Memory Combinations"])

    combos = d["Memory Combinations"].astype(str).tolist()
    gain = (d["MoE-FT"] - d["FT"]).tolist()

    x = list(range(len(combos)))

    plt.figure(figsize=(9.6, 4.6))
    plt.bar(x, gain)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)

    plt.xticks(x, combos)
    plt.xlabel("Memory combination")
    plt.ylabel(f"Gain (MoE-FT − FT) in {metric} (avg)")
    plt.title(f"{subplot} MoE gating gain over FT at WB4 ({metric})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    if annotate:
        for i, v in enumerate(gain):
            if pd.isna(v):
                continue
            plt.text(i, v, f"{v:+.2f}",
                     ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png} (dpi={dpi})")

if __name__ == "__main__":
    CSV_PATH = "figure2.csv"  # <-- change this

    # # Figure 2
    # plot_figure2_grouped_bars(CSV_PATH, metric="F1",
    #                           out_png="Figure2_WB4_groupedbars_F1.png", dpi=1200)
    # plot_figure2_grouped_bars(CSV_PATH, metric="EM",
    #                           out_png="Figure2_WB4_groupedbars_EM.png", dpi=1200)

    # Figure 3
    plot_figure3_moe_gain(CSV_PATH, metric="F1", subplot="(a)",
                          out_png="Figure3_WB4_MoE_minus_FT_gain_F1.png", dpi=1200)
    plot_figure3_moe_gain(CSV_PATH, metric="EM", subplot="(b)",
                          out_png="Figure3_WB4_MoE_minus_FT_gain_EM.png", dpi=1200)
