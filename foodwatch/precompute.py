"""Runs the FoodWatch notebook's staple-series pipeline and saves results.json.
Run once:  python precompute.py   (takes about a minute)"""
import json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore")

STAPLES = ["Maize flour", "Beans (dry)", "Rice", "Sugar", "Wheat flour"]
KEYS = ["commodity", "unit", "market"]
HORIZON, STABLE_BAND = 6, 3.0   # months ahead; +/-3% counts as "stable"

raw = pd.read_csv("data/wfp_food_prices_ken.csv")
raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
raw["price"] = pd.to_numeric(raw["price"], errors="coerce")

food = raw[raw.pricetype == "Retail"].dropna(subset=["date", "market", "commodity", "unit", "price"])
food = food[food.price > 0].copy()
food["month"] = food.date.dt.to_period("M").dt.to_timestamp()
latest = food.month.max()

cov = food.groupby(KEYS)["month"].agg(obs="nunique", first="min", last="max").reset_index()
ok = cov[cov.commodity.isin(STAPLES) & (cov.unit == "KG") & (cov.obs >= 36)
         & (cov["last"] >= latest - pd.DateOffset(months=3))]
score = ok.groupby("market").agg(n=("commodity", "nunique"), obs=("obs", "sum"))
markets = score[score.n == len(STAPLES)].sort_values("obs", ascending=False).head(3).index
sel = ok[ok.market.isin(markets)]

monthly = (food.merge(sel[KEYS], on=KEYS).groupby(["month"] + KEYS)["price"].median().reset_index())
monthly["series"] = monthly.commodity + " | " + monthly.market
wide = monthly.pivot(index="month", columns="series", values="price").sort_index().asfreq("MS")

def latest_run(s, gap=3):
    f, l = s.first_valid_index(), s.last_valid_index()
    s = s.interpolate(method="time", limit=gap, limit_area="inside")
    nan = np.where(s.loc[f:l].isna().to_numpy())[0]
    if len(nan): s.loc[:s.loc[f:l].index[nan[-1]]] = np.nan
    return s
clean = wide.apply(latest_run)

def fit_predict(name, train, steps):
    if name == "Naive": return np.repeat(train.iloc[-1], steps)
    if name == "Holt-Winters":
        m = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,
                                 initialization_method="estimated").fit()
        return np.asarray(m.forecast(steps))
    return np.asarray(ARIMA(train, order=(1, 1, 1)).fit().forecast(steps))

out = {"series": {}, "markets": list(markets), "commodities": STAPLES, "stats": {}}
for col in clean.columns:
    s = clean[col].dropna().asfreq("MS")
    cut = int(len(s) * 0.8)
    train, test = s.iloc[:cut], s.iloc[cut:]
    scores = {}
    for name in ["Naive", "Holt-Winters", "ARIMA(1,1,1)"]:
        try: scores[name] = float(np.sqrt(np.mean((test.values - fit_predict(name, train, len(test))) ** 2)))
        except Exception: pass
    best = min(scores, key=scores.get); rmse = scores[best]
    fc = fit_predict(best, s, HORIZON)
    fdates = pd.date_range(s.index[-1] + pd.DateOffset(months=1), periods=HORIZON, freq="MS")
    now = float(s.iloc[-1]); chg6 = (fc[-1] / now - 1) * 100; chg1 = (fc[0] / now - 1) * 100
    direction = "up" if chg6 > STABLE_BAND else "down" if chg6 < -STABLE_BAND else "stable"

    pct = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna() * 100
    iso = IsolationForest(contamination=0.05, random_state=42).fit(pct.to_frame())
    flag = iso.predict(pct.to_frame()) == -1
    anomalies = [{"date": d.strftime("%Y-%m"), "change": round(float(p), 1), "price": round(float(s[d]), 1)}
                 for d, p, f in zip(pct.index, pct.values, flag) if f]

    hist = s.iloc[-60:]
    commodity, market = col.split(" | ")
    out["series"][col] = {
        "commodity": commodity, "market": market, "model": best, "rmse": round(rmse, 2),
        "now": round(now, 1), "last_month": s.index[-1].strftime("%Y-%m"),
        "direction": direction, "change_1m": round(float(chg1), 1), "change_6m": round(float(chg6), 1),
        "history": {"labels": [d.strftime("%Y-%m") for d in hist.index], "values": [round(float(v), 1) for v in hist.values]},
        "forecast": {"labels": [d.strftime("%Y-%m") for d in fdates], "values": [round(float(v), 1) for v in fc],
                     "band": [round(float(rmse * np.sqrt(i + 1)), 1) for i in range(HORIZON)]},
        "anomalies": anomalies,
    }
    print(f"{col:42s} {best:13s} {direction:6s} {chg6:+.1f}%  anomalies={len(anomalies)}")

out["stats"] = {"rows": int(len(raw)), "markets": int(raw.market.nunique()),
                "commodities": int(raw.commodity.nunique()),
                "from": raw.date.min().strftime("%Y"), "to": raw.date.max().strftime("%Y-%m")}
json.dump(out, open("data/results.json", "w"))
print("saved data/results.json")
