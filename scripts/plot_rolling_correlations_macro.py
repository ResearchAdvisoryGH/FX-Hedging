# plot_rolling_correlations_macro.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\data\FX-Hedging-main")

INPUT_FILE = BASE_DIR / "outputs" / "charts" / "normalized_macro_factor_map_5y.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 90  # trading days, roughly 3-4 months

PAIRS = {
    "Brent vs USD/NGN": ("Brent Crude", "USD/NGN"),
    "Copper vs USD/ZMW": ("Copper", "USD/ZMW"),
    "Cocoa vs USD/GHS": ("Cocoa", "USD/GHS"),
    "Gold vs USD/NGN": ("Gold", "USD/NGN"),
    "Gold vs USD/GHS": ("Gold", "USD/GHS"),
    "Gold vs USD/ZMW": ("Gold", "USD/ZMW"),
}

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

# First column should be Date from the saved normalized CSV index
df = df.rename(columns={df.columns[0]: "Date"})
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

# =========================================================
# COMPUTE LOG RETURNS
# =========================================================

returns = np.log(df / df.shift(1)).dropna()

# =========================================================
# ROLLING CORRELATIONS
# =========================================================

rolling_corrs = pd.DataFrame(index=returns.index)

for label, (x, y) in PAIRS.items():
    if x not in returns.columns or y not in returns.columns:
        print(f"Skipping {label}: missing {x} or {y}")
        continue

    rolling_corrs[label] = returns[x].rolling(WINDOW).corr(returns[y])

# Save rolling correlation data
out_csv = OUTPUT_DIR / f"rolling_correlations_{WINDOW}d_macro_factor_map.csv"
rolling_corrs.to_csv(out_csv)

# =========================================================
# PLOT ALL ROLLING CORRELATIONS
# =========================================================

plt.figure(figsize=(15, 9))

for col in rolling_corrs.columns:
    plt.plot(rolling_corrs.index, rolling_corrs[col], label=col, linewidth=2)

plt.axhline(0, linewidth=1)
plt.axhline(0.5, linewidth=1, linestyle="--")
plt.axhline(-0.5, linewidth=1, linestyle="--")

plt.title(f"{WINDOW}-Day Rolling Correlations: Commodities vs African FX", fontsize=16, pad=15)
plt.ylabel("Rolling Correlation of Daily Log Returns")
plt.xlabel("Date")

plt.ylim(-1, 1)
plt.legend(ncol=2)
plt.tight_layout()

out_png = OUTPUT_DIR / f"rolling_correlations_{WINDOW}d_macro_factor_map.png"
plt.savefig(out_png, dpi=300)

print(f"Saved chart: {out_png}")
print(f"Saved data: {out_csv}")

plt.show()

# =========================================================
# OPTIONAL: PRINT LATEST VALUES
# =========================================================

latest = rolling_corrs.dropna().iloc[-1].sort_values(ascending=False)

print("\nLatest rolling correlations:")
print(latest.to_string())