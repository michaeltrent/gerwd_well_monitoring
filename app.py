
import base64
import io
import os
from datetime import datetime

import dash
from dash import Dash, html, dcc, Input, Output, State, MATCH, ALL, ctx
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import pandas as pd
import plotly.graph_objects as go

# --- Configuration ---
DEFAULT_CSV_PATH = os.environ.get("GVW_CSV_PATH", "/mnt/data/well_data_df_10.07.2025.csv")
AQUIFER_COLORS = {"Dawson": "#e74c3c", "Denver": "#3498db"}

# --- Data utilities ---
REQUIRED_COLS = ["Well_Name", "Latitude", "Longitude", "Aquifer", "Date", "Depth_ft", "Method"]

def coerce_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure types and required columns exist; return cleaned DataFrame."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    out = df.copy()
    # Strip whitespace in string columns
    for col in ["Well_Name", "Aquifer", "Method"]:
        out[col] = out[col].astype(str).str.strip()

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

def read_default_csv():
    if os.path.exists(DEFAULT_CSV_PATH):
        try:
            df = pd.read_csv(DEFAULT_CSV_PATH)
            return coerce_dataframe(df)
        except Exception as e:
            print(f"Failed to read default CSV: {e}")
    # fallback to empty structure
    return pd.DataFrame(columns=REQUIRED_COLS)

def parse_contents(contents, filename):
    """Parse uploaded file contents into a DataFrame."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        return coerce_dataframe(df)
    except Exception as e:
        raise ValueError(f"Error processing file '{filename}': {e}")

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
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Depth to Water (feet)"
    )
    # Reverse Y so deeper water (bigger number) is plotted lower
    fig.update_yaxes(autorange="reversed")
    return fig

def fig_aquifer_averages(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    g = (
        df.groupby(["Date", "Aquifer"])["Depth_ft"]
        .mean()
        .reset_index()
        .pivot(index="Date", columns="Aquifer", values="Depth_ft")
        .sort_index()
    )

    fig = go.Figure()
    for aquifer in g.columns:
        color = AQUIFER_COLORS.get(aquifer, None)
        fig.add_trace(go.Scatter(
            x=g.index,
            y=g[aquifer],
            mode="lines+markers",
            name=f"{aquifer} Average",
            line=dict(width=3, color=color) if color else dict(width=3),
            marker=dict(size=7)
        ))
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
    return (df["Latitude"].mean(), df["Longitude"].mean())

# --- App & Layout ---
external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # for gunicorn / production

initial_df = read_default_csv()

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("🏞️ Grandview Estates Water Well Monitor", className="mt-3 mb-0"),
            html.Div("Rural Water Conservation District — Real-time Well Depth Analysis", className="text-muted mb-3"),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📤 Data — Upload CSV / Excel", className="fw-semibold"),
                dbc.CardBody([
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(["Drag and drop or ", html.A("select a CSV/Excel file")]),
                        style={
                            "width": "100%", "height": "80px", "lineHeight": "80px",
                            "borderWidth": "2px", "borderStyle": "dashed",
                            "borderRadius": "8px", "textAlign": "center",
                            "background": "#e3f2fd"
                        },
                        multiple=False
                    ),
                    html.Small("Expected columns: Well_Name, Latitude, Longitude, Aquifer, Date, Depth_ft, Method", className="text-muted"),
                    html.Div(id="upload-status", className="mt-2"),
                    dcc.Store(id="data-store", data=initial_df.to_dict("records")),
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
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="well-select",
                                options=[{"label": w, "value": w} for w in sorted(initial_df["Well_Name"].unique())],
                                placeholder="Select a well...",
                                persistence=True,
                            )
                        ], md=8),
                        dbc.Col([
                            dbc.Alert(id="well-meta", color="light", is_open=False, className="mb-0",
                                      style={"borderLeft": "4px solid #667eea"}),
                        ], md=4)
                    ], className="g-2"),

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
    ]),

    html.Div(id="hidden-div")  # placeholder for pattern-matching inputs
], fluid=True)

# --- Callbacks ---
@app.callback(
    Output("upload-status", "children"),
    Output("data-store", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate
    try:
        df = parse_contents(contents, filename)
        msg = dbc.Alert([
            html.Span("Loaded "), html.B(filename), html.Span(f" — {len(df):,} rows")
        ], color="success", className="mt-2")
        return msg, df.to_dict("records")
    except Exception as e:
        msg = dbc.Alert(str(e), color="danger", className="mt-2")
        return msg, dash.no_update

@app.callback(
    Output("markers-layer", "children"),
    Output("map", "center"),
    Input("data-store", "data"),
)
def refresh_markers(data):
    df = pd.DataFrame(data) if data else pd.DataFrame(columns=REQUIRED_COLS)
    return build_markers(df), map_center(df)

@app.callback(
    Output("well-select", "options"),
    Input("data-store", "data"),
)
def refresh_well_options(data):
    df = pd.DataFrame(data) if data else pd.DataFrame(columns=REQUIRED_COLS)
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
    Output("well-meta", "children"),
    Output("well-meta", "is_open"),
    Input("data-store", "data"),
    Input("well-select", "value"),
)
def update_time_graph(data, well):
    df = pd.DataFrame(data) if data else pd.DataFrame(columns=REQUIRED_COLS)
    fig = fig_time_series(df, well)

    if not well or df.empty or fig is None or len(fig.data) == 0:
        return go.Figure(), "", False

    sdf = df[df["Well_Name"] == well].sort_values("Date")
    last = sdf.iloc[-1]
    meta = [
        html.Div([html.B("Well: "), html.Span(str(last["Well_Name"]))]),
        html.Div([html.B("Aquifer: "), html.Span(str(last["Aquifer"]))]),
        html.Div([html.B("Date: "), html.Span(last["Date"].strftime("%Y-%m-%d"))]),
        html.Div([html.B("Method: "), html.Span(str(last["Method"]))]),
    ]
    return fig, meta, True

@app.callback(
    Output("aquifer-avg", "figure"),
    Input("data-store", "data"),
)
def update_aquifer_avg(data):
    df = pd.DataFrame(data) if data else pd.DataFrame(columns=REQUIRED_COLS)
    return fig_aquifer_averages(df)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
