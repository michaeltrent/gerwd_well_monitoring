
import os

import dash
from dash import Dash, html, dcc, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import erfc, sqrt

# --- Configuration ---
DEFAULT_CSV_PATH = os.environ.get(
    "GVW_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "well_data_df_10.07.2025.csv"),
)
AQUIFER_COLORS = {"Dawson Arkose": "#e74c3c", "Denver Formation": "#3498db"}


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a HEX color string to an rgba() string with the provided alpha."""

    if not hex_color:
        return f"rgba(102, 102, 102, {alpha})"
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return f"rgba(102, 102, 102, {alpha})"
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

# --- Data utilities ---
REQUIRED_COLS = ["Well_Name", "Latitude", "Longitude", "Aquifer", "Date", "Depth_ft", "Method"]
ESSENTIAL_COLS = ["Well_Name", "Latitude", "Longitude", "Aquifer", "Date", "Depth_ft"]

COLUMN_ALIASES = {
    "well_name": "Well_Name",
    "well": "Well_Name",
    "local_aquafer": "Aquifer",
    "local_aquifer": "Aquifer",
    "aquifer": "Aquifer",
    "phenomenontime": "Date",
    "date": "Date",
    "result": "Depth_ft",
    "depth_ft": "Depth_ft",
    "water level": "Depth_ft",
    "resultmethod": "Method",
    "method": "Method",
    "latitude": "Latitude",
    "lat": "Latitude",
    "longitude": "Longitude",
    "lon": "Longitude",
    "long": "Longitude",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column variants to the expected schema."""

    rename_map = {}
    for col in df.columns:
        key = str(col).strip()
        lowered = key.lower()
        if lowered.startswith("unnamed"):
            continue
        target = COLUMN_ALIASES.get(lowered)
        if target:
            rename_map[col] = target
    out = df.rename(columns=rename_map)
    out = out.loc[:, [c for c in out.columns if not str(c).lower().startswith("unnamed")]]

    if "Method" not in out.columns:
        out["Method"] = "Unknown"

    return out

def coerce_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure types and required columns exist; return cleaned DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    out = standardize_columns(df)

    missing = [c for c in ESSENTIAL_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(out.columns)}")

    out = out.copy()
    # Strip whitespace in string columns
    for col in ["Well_Name", "Aquifer", "Method"]:
        out[col] = out[col].astype(str).str.strip()

    if "Method" in out.columns:
        out["Method"] = out["Method"].replace({"nan": "Unknown", "None": "Unknown"}).replace("", "Unknown")

    # Parse numerics
    out["Latitude"] = pd.to_numeric(out["Latitude"], errors="coerce")
    out["Longitude"] = pd.to_numeric(out["Longitude"], errors="coerce")
    out["Depth_ft"] = pd.to_numeric(out["Depth_ft"], errors="coerce")

    # Parse dates
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    # Drop rows with missing critical fields
    out = out.dropna(subset=["Well_Name", "Latitude", "Longitude", "Aquifer", "Date", "Depth_ft"])

    # Sort for consistency
    out = out.sort_values(["Well_Name", "Date"]).reset_index(drop=True)
    return out


def dataframe_from_store(data) -> pd.DataFrame:
    """Rebuild a cleaned DataFrame from serialized dcc.Store data."""

    if not data:
        return pd.DataFrame(columns=REQUIRED_COLS)

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    try:
        return coerce_dataframe(df)
    except ValueError:
        # Fall back to lightweight parsing while preserving available rows.
        df = standardize_columns(df)
        for col in ["Latitude", "Longitude", "Depth_ft"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df

def read_default_csv():
    if os.path.exists(DEFAULT_CSV_PATH):
        try:
            df = pd.read_csv(DEFAULT_CSV_PATH)
            return coerce_dataframe(df)
        except Exception as e:
            print(f"Failed to read default CSV: {e}")
    # fallback to empty structure
    return pd.DataFrame(columns=REQUIRED_COLS)


# --- Statistics utilities ---
def _normal_approx_pvalue(t_stat: float) -> float:
    """Two-sided p-value using a normal approximation for large df."""

    return 2 * 0.5 * erfc(abs(t_stat) / sqrt(2))


def linear_regression_stats(x: np.ndarray, y: np.ndarray):
    """Return slope/intercept stats for simple linear regression.

    Parameters
    ----------
    x, y: array-like
        Numeric vectors with matching length. NaN/inf values should be removed
        prior to calling this function.

    Returns
    -------
    dict | None
        Keys include slope, intercept, r_squared, p_value, slope_stderr, and
        y_hat for the provided x values. Returns None when computation is not
        possible (e.g., fewer than two finite samples).
    """

    if x.size < 2 or y.size < 2:
        return None

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None

    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    residuals = y - y_hat

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = x.size
    slope_stderr = np.nan
    t_stat = np.nan
    p_value = None

    if n > 2:
        s_err = np.sqrt(ss_res / (n - 2))
        ssx = np.sum((x - x.mean()) ** 2)
        if ssx > 0:
            slope_stderr = s_err / np.sqrt(ssx)
            if slope_stderr > 0:
                t_stat = slope / slope_stderr
                df = n - 2
                try:
                    from scipy import stats  # type: ignore

                    p_value = 2 * stats.t.sf(np.abs(t_stat), df)
                except Exception:
                    p_value = _normal_approx_pvalue(t_stat)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": float(r_squared),
        "p_value": p_value,
        "slope_stderr": float(slope_stderr) if np.isfinite(slope_stderr) else None,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else None,
        "y_hat": y_hat,
        "x": x,
    }


# --- Build figures ---
def fig_time_series(df: pd.DataFrame, well: str) -> go.Figure:
    if not well or df.empty:
        return go.Figure()

    sdf = df[df["Well_Name"] == well].sort_values("Date")
    if sdf.empty:
        return go.Figure()

    aquifer = sdf["Aquifer"].iloc[-1]
    color = AQUIFER_COLORS.get(aquifer, "#666")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sdf["Date"],
        y=sdf["Depth_ft"],
        mode="lines+markers",
        name="Depth to Water (ft)",
        line=dict(width=3, color=color),
        marker=dict(size=8)
    ))

    base_date = sdf["Date"].min()
    x_days = (sdf["Date"] - base_date).dt.total_seconds() / 86400.0
    y_depth = sdf["Depth_ft"].astype(float)
    reg = linear_regression_stats(x_days.to_numpy(dtype=float), y_depth.to_numpy(dtype=float))

    annotation_text = None
    if reg:
        trend_y = reg["slope"] * x_days + reg["intercept"]
        fig.add_trace(go.Scatter(
            x=sdf["Date"],
            y=trend_y,
            mode="lines",
            name="Trend line",
            line=dict(color=color, width=2, dash="dash")
        ))

        slope_daily = reg["slope"]
        slope_yearly = slope_daily * 365.25
        p_value = reg["p_value"]
        significance = None
        if p_value is not None:
            significance = "Yes" if p_value < 0.05 else "No"

        annotation_lines = [
            f"Depth = {reg['intercept']:.2f} + {slope_daily:.4f} × days since {base_date.date()}",
            f"R² = {reg['r_squared']:.3f}",
            f"Slope ≈ {slope_yearly:.2f} ft/yr",
        ]
        if p_value is not None:
            annotation_lines.append(f"Slope p = {p_value:.3g} → significant? {significance}")
        else:
            annotation_lines.append("Slope significance unavailable")
        annotation_text = "<br>".join(annotation_lines)

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Depth to Water (feet)"
    )

    if annotation_text:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            text=annotation_text,
            align="left",
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor=color,
            borderwidth=1,
        )
    # Reverse Y so deeper water (bigger number) is plotted lower
    fig.update_yaxes(autorange="reversed")
    return fig

def fig_aquifer_averages(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    monthly = df.copy()
    monthly["Date"] = monthly["Date"].dt.to_period("M").dt.to_timestamp()
    stats = (
        monthly.groupby(["Aquifer", "Date"])["Depth_ft"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    stats = stats.sort_values(["Aquifer", "Date"])  # chronological for each aquifer
    stats["sem"] = stats["std"].div(np.sqrt(stats["count"].replace(0, np.nan)))
    stats["sem"] = stats["sem"].fillna(0)

    fig = go.Figure()
    for aquifer, group in stats.groupby("Aquifer"):
        group = group.sort_values("Date")
        color = AQUIFER_COLORS.get(aquifer, "#666666")

        upper = group["mean"] + group["sem"]
        lower = group["mean"] - group["sem"]
        if group.shape[0] >= 2 and not (upper.equals(lower)):
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([group["Date"], group["Date"].iloc[::-1]]),
                    y=pd.concat([upper, lower.iloc[::-1]]),
                    fill="toself",
                    fillcolor=hex_to_rgba(color, 0.18),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=group["Date"],
                y=group["mean"],
                mode="lines+markers",
                name=f"{aquifer} Average",
                line=dict(width=3, color=color),
                marker=dict(size=7)
            )
        )

        if group.shape[0] >= 2:
            base_date = group["Date"].min()
            x_days = (group["Date"] - base_date).dt.total_seconds() / 86400.0
            reg = linear_regression_stats(
                x_days.to_numpy(dtype=float),
                group["mean"].to_numpy(dtype=float)
            )
            if reg:
                trend = reg["slope"] * x_days + reg["intercept"]
                fig.add_trace(
                    go.Scatter(
                        x=group["Date"],
                        y=trend,
                        mode="lines",
                        name=f"{aquifer} Trend",
                        line=dict(color=color, width=2, dash="dot")
                    )
                )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Average Depth to Water (feet)",
    )
    fig.update_yaxes(autorange="reversed")
    return fig

# --- Build map layer ---
def build_markers(df: pd.DataFrame):
    """Return a list of dl.Marker for unique wells (latest record per well)."""
    if df.empty:
        return []

    # latest record per well (so popup shows current meta)
    latest = df.sort_values("Date").groupby("Well_Name").tail(1)

    markers = []
    for _, row in latest.iterrows():
        aquifer = str(row["Aquifer"])
        color = AQUIFER_COLORS.get(aquifer, "#666")
        well = row["Well_Name"]
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])

        popup = dl.Popup([
            html.B(well),
            html.Br(),
            html.Span(f"Aquifer: {aquifer}"),
            html.Br(),
            html.Span(f"Location: {lat:.4f}, {lon:.4f}"),
            html.Br(),
            html.Span("Click marker to select ▶"),
        ])

        markers.append(
            dl.CircleMarker(
                id={"type": "well-marker", "well": well},
                center=(lat, lon),
                radius=10,
                color="#ffffff",
                weight=2,
                fillColor=color,
                fillOpacity=0.9,
                children=[popup],
                n_clicks=0,
            )
        )
    return markers

def map_center(df: pd.DataFrame):
    if df.empty:
        return (39.545146, -104.820199)
    coords = df.dropna(subset=["Latitude", "Longitude"])
    if coords.empty:
        return (39.545146, -104.820199)
    return (coords["Latitude"].mean(), coords["Longitude"].mean())

# --- App & Layout ---
external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # for gunicorn / production

initial_df = read_default_csv()
initial_records = initial_df.to_dict("records")
initial_aquifers = (
    sorted(initial_df["Aquifer"].dropna().unique()) if not initial_df.empty else []
)

if not initial_df.empty and initial_df["Date"].notna().any():
    min_date = initial_df["Date"].min()
    max_date = initial_df["Date"].max()
else:
    min_date = max_date = None

app.layout = dbc.Container([
    dcc.Store(id="data-store", data=initial_records),

    dbc.Row([
        dbc.Col([
            html.H2("🏞️ Grandview Estates Water Well Monitor", className="mt-3 mb-0"),
            html.Div("Rural Water Conservation District — Real-time Well Depth Analysis", className="text-muted mb-3"),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📁 Data Source", className="fw-semibold"),
                dbc.CardBody([
                    html.P("Historical well measurements are loaded automatically from the bundled dataset."),
                    html.Ul([
                        html.Li(html.Span([html.B("File:"), f" {os.path.basename(DEFAULT_CSV_PATH)}"])),
                        html.Li(html.Span([html.B("Records:"), f" {len(initial_df):,}" if not initial_df.empty else " 0"])),
                        html.Li(
                            html.Span([
                                html.B("Aquifers:"),
                                " " + ", ".join(initial_aquifers) if initial_aquifers else " N/A",
                            ])
                        ),
                        html.Li(
                            html.Span([
                                html.B("Date Range:"),
                                " "
                                + (
                                    f"{min_date.date()} → {max_date.date()}"
                                    if min_date is not None and max_date is not None
                                    else "N/A"
                                ),
                            ])
                        ),
                    ], className="mb-0"),
                ])
            ])
        ], width=12)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📍 Well Locations Map", className="fw-semibold"),
                dbc.CardBody([
                    dl.Map(
                        id="map",
                        center=map_center(initial_df),
                        zoom=13,
                        style={"width": "100%", "height": "420px", "borderRadius": "8px", "border": "2px solid #e9ecef"},
                        children=[
                            dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                                         attribution="© OpenStreetMap contributors"),
                            dl.FeatureGroup(id="markers-layer", children=build_markers(initial_df)),
                        ],
                    )
                ])
            ])
        ], md=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Well Depth Over Time", className="fw-semibold"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="well-select",
                        options=[{"label": w, "value": w} for w in sorted(initial_df["Well_Name"].unique())],
                        placeholder="Select a well...",
                        persistence=True,
                        className="mb-3"
                    ),

                    dcc.Graph(id="time-series", style={"height": "320px"}),
                ])
            ])
        ], md=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🌊 Average Depth by Aquifer", className="fw-semibold"),
                dbc.CardBody([
                    dcc.Graph(id="aquifer-avg", style={"height": "340px"})
                ])
            ])
        ], width=12)
    ])
], fluid=True)

# --- Callbacks ---
@app.callback(
    Output("markers-layer", "children"),
    Output("map", "center"),
    Input("data-store", "data"),
)
def refresh_markers(data):
    df = dataframe_from_store(data)
    return build_markers(df), map_center(df)

@app.callback(
    Output("well-select", "options"),
    Input("data-store", "data"),
)
def refresh_well_options(data):
    df = dataframe_from_store(data)
    wells = sorted(df["Well_Name"].unique()) if not df.empty else []
    return [{"label": w, "value": w} for w in wells]

# Marker click -> set dropdown
@app.callback(
    Output("well-select", "value"),
    Input({"type": "well-marker", "well": ALL}, "n_clicks"),
    State({"type": "well-marker", "well": ALL}, "id"),
    prevent_initial_call=True,
)
def marker_click_to_dropdown(n_clicks, ids):
    if not n_clicks or not ids:
        raise dash.exceptions.PreventUpdate
    # pick the most recently clicked marker
    if ctx.triggered_id is None:
        raise dash.exceptions.PreventUpdate
    if isinstance(ctx.triggered_id, dict) and "well" in ctx.triggered_id:
        return ctx.triggered_id["well"]
    raise dash.exceptions.PreventUpdate

@app.callback(
    Output("time-series", "figure"),
    Input("data-store", "data"),
    Input("well-select", "value"),
)
def update_time_graph(data, well):
    df = dataframe_from_store(data)
    return fig_time_series(df, well)

@app.callback(
    Output("aquifer-avg", "figure"),
    Input("data-store", "data"),
)
def update_aquifer_avg(data):
    df = dataframe_from_store(data)
    return fig_aquifer_averages(df)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
