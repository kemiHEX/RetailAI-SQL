from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from retail_ai import db, ml

ROOT = Path(__file__).resolve().parent

store_df = db.stores()
product_df = db.products()
STORE_CHOICES = {f"{r.store_name} — {r.city}": int(r.store_id) for r in store_df.itertuples()}
PRODUCT_CHOICES = {f"{r.product_name} ({r.category})": int(r.product_id) for r in product_df.itertuples()}
STORE_LABELS = {v: k for k, v in STORE_CHOICES.items()}
PRODUCT_LABELS = {v: k for k, v in PRODUCT_CHOICES.items()}


def make_line_plot(x, y, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    return fig


def dashboard():
    k = db.dashboard_kpis()
    monthly = db.monthly_revenue()
    categories = db.category_revenue()
    top = db.top_products()

    fig1 = make_line_plot(monthly["month"], monthly["revenue"], "Monthly Revenue", "Revenue ($)")
    fig2, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(categories["category"], categories["revenue"])
    ax.set_title("Revenue by Category")
    ax.set_ylabel("Revenue ($)")
    ax.tick_params(axis="x", rotation=35)
    fig2.tight_layout()
    summary = (
        f"### Retail overview\n"
        f"**${k['revenue']:,.0f} revenue** · **{int(k['units']):,} units sold** · "
        f"**{k['stockout_rate']:.2f}% recorded stockouts** · **{int(k['products'])} products** · "
        f"**{int(k['stores'])} stores**  \n"
        f"Dataset window: **{k['min_date']} → {k['max_date']}**"
    )
    return summary, fig1, fig2, top


def run_forecast(store_label, product_label, horizon, promo):
    sid = STORE_CHOICES[store_label]
    pid = PRODUCT_CHOICES[product_label]
    forecast, history = ml.forecast_demand(sid, pid, int(horizon), bool(promo))
    run_id = db.save_forecast(sid, pid, int(horizon), bool(promo), forecast)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(pd.to_datetime(history["sale_date"]), history["units_sold"], label="Actual history")
    ax.plot(pd.to_datetime(forecast["forecast_date"]), forecast["predicted_units"], marker="o", label="ML forecast")
    ax.set_title(f"Demand Forecast — {product_label} @ {store_label}")
    ax.set_ylabel("Units")
    ax.legend()
    fig.tight_layout()
    total = forecast["predicted_units"].sum()
    status = f"Saved forecast run **#{run_id}** to SQLite · predicted **{total:.1f} units** over {int(horizon)} days."
    return fig, forecast, status, db.recent_forecast_runs()


def risk_view(store_label):
    sid = None if store_label == "All stores" else STORE_CHOICES[store_label]
    df = ml.inventory_risk_table(sid)
    if df.empty:
        return df, "No risk records available."
    high = int((df["Risk"] == "HIGH").sum())
    medium = int((df["Risk"] == "MEDIUM").sum())
    return df, f"**{high} high-risk** and **{medium} medium-risk** store/product combinations based on the trained classifier."


def preset_sql(name):
    return db.PRESET_QUERIES.get(name, "")


def execute_sql(sql):
    try:
        df = db.run_readonly_sql(sql)
        return df, f"Query returned **{len(df):,} rows**."
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]}), f"SQL error: `{e}`"


def model_results():
    meta = ml.metadata()
    reg = pd.DataFrame(meta["regression"]["candidates"])
    clf = pd.DataFrame(meta["classification"]["candidates"])
    imp = pd.read_csv(ROOT / "data" / "feature_importance.csv")
    selected_reg = meta["regression"]["selected_model"]
    selected_clf = meta["classification"]["selected_model"]
    r = reg[reg["model"] == selected_reg].iloc[0]
    c = clf[clf["model"] == selected_clf].iloc[0]
    text = (
        f"### Trained model results\n"
        f"Demand model: **{selected_reg}** · MAE **{r.mae:.2f} units** · RMSE **{r.rmse:.2f}** · "
        f"R² **{r.r2:.3f}** · MAPE **{r.mape_pct:.1f}%**  \n"
        f"Inventory-risk model: **{selected_clf}** · F1 **{c.f1:.3f}** · ROC-AUC **{c.roc_auc:.3f}**  \n"
        f"Chronological holdout begins **{meta['split_date']}** ({meta['test_rows']:,} test rows)."
    )
    return text, reg, clf, imp


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto !important;}
.hero {text-align:center; padding: 8px 0 16px 0;}
.small-note {opacity:0.78; font-size:0.92rem;}
"""

with gr.Blocks(title="RetailAI SQL — Demand & Inventory Intelligence") as demo:
    gr.Markdown("""
    <div class='hero'>
      <h1>RetailAI SQL</h1>
      <p>Demand forecasting, inventory-risk prediction and SQL analytics in one local ML application.</p>
    </div>
    """)

    with gr.Tab("Executive Dashboard"):
        dash_summary = gr.Markdown()
        with gr.Row():
            rev_plot = gr.Plot()
            cat_plot = gr.Plot()
        top_table = gr.Dataframe(interactive=False)
        refresh_dash = gr.Button("Refresh dashboard")
        refresh_dash.click(dashboard, outputs=[dash_summary, rev_plot, cat_plot, top_table])

    with gr.Tab("Demand Forecast"):
        gr.Markdown("Forecast future item demand using a model trained on lag, rolling-demand, calendar, store and promotion features.")
        with gr.Row():
            store = gr.Dropdown(list(STORE_CHOICES), value=list(STORE_CHOICES)[0], label="Store")
            product = gr.Dropdown(list(PRODUCT_CHOICES), value=list(PRODUCT_CHOICES)[0], label="Product")
            horizon = gr.Slider(7, 30, value=14, step=1, label="Forecast horizon (days)")
            promo = gr.Checkbox(value=False, label="Promotion scenario")
        forecast_btn = gr.Button("Generate & save forecast", variant="primary")
        forecast_status = gr.Markdown()
        forecast_plot = gr.Plot()
        with gr.Row():
            forecast_table = gr.Dataframe(interactive=False)
            recent_runs = gr.Dataframe(value=db.recent_forecast_runs(), interactive=False, label="Recent SQL forecast runs")
        forecast_btn.click(run_forecast, [store, product, horizon, promo], [forecast_plot, forecast_table, forecast_status, recent_runs])

    with gr.Tab("Inventory Risk"):
        with gr.Row():
            risk_store = gr.Dropdown(["All stores"] + list(STORE_CHOICES), value="All stores", label="Store filter")
            risk_btn = gr.Button("Score inventory risk", variant="primary")
        risk_summary = gr.Markdown()
        risk_table = gr.Dataframe(interactive=False)
        risk_btn.click(risk_view, [risk_store], [risk_table, risk_summary])

    with gr.Tab("SQL Explorer"):
        gr.Markdown("Run **read-only SQL** against the same SQLite database that powers the ML app. Presets demonstrate joins, aggregates, CTEs and window functions.")
        preset = gr.Dropdown(list(db.PRESET_QUERIES), value=list(db.PRESET_QUERIES)[0], label="Example query")
        sql_box = gr.Code(value=db.PRESET_QUERIES[list(db.PRESET_QUERIES)[0]], language="sql", label="SQL")
        preset.change(preset_sql, [preset], [sql_box])
        sql_btn = gr.Button("Run query", variant="primary")
        sql_status = gr.Markdown()
        sql_result = gr.Dataframe(interactive=False)
        sql_btn.click(execute_sql, [sql_box], [sql_result, sql_status])

    with gr.Tab("Model & Data"):
        result_text = gr.Markdown()
        with gr.Row():
            reg_table = gr.Dataframe(interactive=False, label="Demand-model comparison")
            clf_table = gr.Dataframe(interactive=False, label="Inventory-risk model comparison")
        imp_table = gr.Dataframe(interactive=False, label="Permutation feature importance")
        gr.Markdown("""
        **Dataset note:** the repository ships with a deterministic synthetic retail dataset so the project is fully reproducible and contains no proprietary customer data. It models promotions, weekly/annual seasonality, category effects, store-level demand, replenishment and supply delays.
        """, elem_classes=["small-note"])
        model_btn = gr.Button("Load training results")
        model_btn.click(model_results, outputs=[result_text, reg_table, clf_table, imp_table])

    demo.load(dashboard, outputs=[dash_summary, rev_plot, cat_plot, top_table])
    demo.load(model_results, outputs=[result_text, reg_table, clf_table, imp_table])

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=CSS)
