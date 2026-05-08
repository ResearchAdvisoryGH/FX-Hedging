# plot_normalized_macro_factor_map.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\data\FX-Hedging-main")

FILES = {
    # Commodities
    "Brent Crude": BASE_DIR / "Commodities" / "Global Stats" / "Brent Crude" / "brent_crude_5y_daily.csv",
    "Copper": BASE_DIR / "Commodities" / "Global Stats" / "Copper" / "copper_5y_daily.csv",
    "Cocoa": BASE_DIR / "Commodities" / "Global Stats" / "Cocoa" / "cocoa_5y_daily.csv",
    "Gold": BASE_DIR / "Commodities" / "Global Stats" / "Gold" / "gold_5y_daily.csv",

    # FX rates
    "USD/NGN": BASE_DIR / "FX Rates (USD - NGN)" / "usdngn_5y_daily.csv",
    "USD/ZMW": BASE_DIR / "FX Rates (USD - ZMW)" / "usdzmw_5y_daily.csv",
    "USD/GHS": BASE_DIR / "FX Rates (USD - GHS)" / "usdghs_5y_daily.csv",
}

OUTPUT_DIR = BASE_DIR / "outputs" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "normalized_macro_factor_map_5y.png"


def load_series(label, path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df[["Date", "Close"]].dropna()
    df = df[df["Close"] > 0]
    df = df.rename(columns={"Close": label})
    return df


merged = None

for label, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file for {label}: {path}")

    temp = load_series(label, path)

    if merged is None:
        merged = temp
    else:
        merged = pd.merge(merged, temp, on="Date", how="outer")


merged = merged.sort_values("Date").set_index("Date")
merged = merged.ffill().dropna()

normalized = merged / merged.iloc[0] * 100

normalized_out = OUTPUT_DIR / "normalized_macro_factor_map_5y.csv"
normalized.to_csv(normalized_out)

plt.figure(figsize=(15, 9))

for col in normalized.columns:
    plt.plot(normalized.index, normalized[col], label=col, linewidth=2)

plt.axhline(100, linewidth=1, linestyle="--")

plt.title("Normalized African FX + Commodity Macro Factor Map (5Y)", fontsize=16, pad=15)
plt.ylabel("Normalized Index Level (Start = 100)")
plt.xlabel("Date")

plt.legend(ncol=2)
plt.tight_layout()

plt.savefig(OUT_FILE, dpi=300)

print(f"Saved chart: {OUT_FILE}")
print(f"Saved normalized data: {normalized_out}")

plt.show()