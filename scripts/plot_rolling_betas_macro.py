# plot_rolling_betas_macro.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\data\FX-Hedging-main")

INPUT_FILE = BASE_DIR / "outputs" / "charts" / "normalized_macro_factor_map_5y.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 90  # trading days

PAIRS = {
    "NGN beta to Brent": ("Brent Crude", "USD/NGN"),
    "ZMW beta to Copper": ("Copper", "USD/ZMW"),
    "GHS beta to Cocoa": ("Cocoa", "USD/GHS"),
    "NGN beta to Gold": ("Gold", "USD/NGN"),
    "GHS beta to Gold": ("Gold", "USD/GHS"),
    "ZMW beta to Gold": ("Gold", "USD/ZMW"),
}


def rolling_beta(y, x, window):
    """
    Rolling beta from regression:
        y_t = alpha + beta * x_t + error_t

    beta = Cov(y, x) / Var(x)
    """
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

df = df.rename(columns={df.columns[0]: "Date"})
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

# =========================================================
# COMPUTE DAILY LOG RETURNS
# =========================================================

returns = np.log(df / df.shift(1)).dropna()

# =========================================================
# COMPUTE ROLLING BETAS
# =========================================================

rolling_betas = pd.DataFrame(index=returns.index)

for label, (x_col, y_col) in PAIRS.items():
    if x_col not in returns.columns or y_col not in returns.columns:
        print(f"Skipping {label}: missing {x_col} or {y_col}")
        continue

    # y = FX return, x = commodity return
    rolling_betas[label] = rolling_beta(
        y=returns[y_col],
        x=returns[x_col],
        window=WINDOW
    )

# Save beta data
out_csv = OUTPUT_DIR / f"rolling_betas_{WINDOW}d_macro_factor_map.csv"
rolling_betas.to_csv(out_csv)

# =========================================================
# PLOT ALL ROLLING BETAS
# =========================================================

plt.figure(figsize=(15, 9))

for col in rolling_betas.columns:
    plt.plot(rolling_betas.index, rolling_betas[col], label=col, linewidth=2)

plt.axhline(0, linewidth=1)
plt.axhline(1, linewidth=1, linestyle="--")
plt.axhline(-1, linewidth=1, linestyle="--")

plt.title(f"{WINDOW}-Day Rolling Betas: African FX Sensitivity to Commodity Shocks", fontsize=16, pad=15)
plt.ylabel("Rolling Beta of FX Returns to Commodity Returns")
plt.xlabel("Date")

plt.legend(ncol=2)
plt.tight_layout()

out_png = OUTPUT_DIR / f"rolling_betas_{WINDOW}d_macro_factor_map.png"
plt.savefig(out_png, dpi=300)

print(f"Saved chart: {out_png}")
print(f"Saved data: {out_csv}")

plt.show()

# =========================================================
# PRINT LATEST VALUES
# =========================================================

latest = rolling_betas.dropna().iloc[-1].sort_values(ascending=False)

print("\nLatest rolling betas:")
print(latest.to_string())