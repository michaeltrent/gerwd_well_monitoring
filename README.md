# Grandview Estates Water Well Monitor (Dash)

A Python-based web dashboard to monitor well depths across the Grandview Estates Rural Water Conservation District.

## Features
- Upload CSV/Excel data (columns required: `Well_Name, Latitude, Longitude, Aquifer, Date, Depth_ft, Method`)
- Interactive map with clickable well markers (Dawson=red, Denver=blue)
- Time series chart per well with reversed Y-axis (deeper water lower)
- Average depth by aquifer over time
- Metadata panel for the latest reading of the selected well

## Quickstart

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8050

## Default Data
This project will try to load a default CSV at:
```
/mnt/data/well_data_df_10.07.2025.csv
```
You can override this path by setting an environment variable before running:
```bash
export GVW_CSV_PATH=/path/to/your/data.csv
# Windows (Powershell)
$env:GVW_CSV_PATH="C:\\path\\to\\your\\data.csv"
```

## Notes
- Date parsing uses pandas; acceptable formats include `YYYY-MM-DD` or typical Excel/CSV date strings.
- Map centers on the mean latitude/longitude of the dataset.
- Marker click selects the well and updates the time series and metadata.
