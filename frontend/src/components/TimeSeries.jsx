import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { useTheme } from '../ThemeContext'
import { fetchTimeseries } from '../api'
import { baseLayout, chartColors } from '../plotlyTheme'

export default function TimeSeries({ wells, selectedWell, onSelectWell, dateRange }) {
  const { dark } = useTheme()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!selectedWell) { setData(null); return }
    setLoading(true)
    setError(null)
    fetchTimeseries(selectedWell, dateRange)
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [selectedWell, dateRange])

  const wellNames = wells.map((w) => w.well_name).sort()
  const c = chartColors(dark)

  const traces = []
  let annotation = null

  if (data) {
    traces.push({
      x: data.dates,
      y: data.depths,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Depth to Water (ft)',
      line: { width: 2.5, color: data.color },
      marker: { size: 6, color: data.color, line: { color: dark ? '#1e293b' : '#fff', width: 1.5 } },
    })

    if (data.trend) {
      const t = data.trend
      traces.push({
        x: t.dates,
        y: t.values,
        type: 'scatter',
        mode: 'lines',
        name: 'Trend',
        line: { color: data.color, width: 1.5, dash: 'dash' },
        opacity: 0.7,
      })

      const pLine =
        t.p_value !== null
          ? `p = ${t.p_value.toFixed(3)}  ·  significant? ${t.p_value < 0.05 ? 'Yes ✓' : 'No'}`
          : 'p-value unavailable'

      annotation = {
        xref: 'paper', yref: 'paper',
        x: 0.01, y: 0.99,
        xanchor: 'left', yanchor: 'top',
        align: 'left',
        font: { size: 11, family: 'Inter, sans-serif', color: c.annotationText },
        bgcolor: c.annotationBg,
        bordercolor: data.color,
        borderwidth: 1.5,
        borderpad: 8,
        text: [
          `<b>${data.well}</b>`,
          `Slope ≈ <b>${t.slope_yearly.toFixed(2)} ft/yr</b>`,
          `R² = ${t.r_squared.toFixed(3)}  ·  ${pLine}`,
        ].join('<br>'),
      }
    }
  }

  const layout = {
    ...baseLayout(dark),
    xaxis: { ...baseLayout(dark).xaxis, title: { ...baseLayout(dark).xaxis.title, text: 'Date' } },
    yaxis: { ...baseLayout(dark).yaxis, title: { ...baseLayout(dark).yaxis.title, text: 'Depth to Water (ft)' }, autorange: 'reversed' },
    annotations: annotation ? [annotation] : [],
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 overflow-hidden flex flex-col">
      {/* Card header */}
      <div className="flex items-center gap-3 px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100 dark:border-slate-700">
        <div className="w-7 h-7 rounded-lg bg-sky-50 dark:bg-sky-900/40 flex items-center justify-center text-base">
          📈
        </div>
        <h2 className="font-semibold text-slate-700 dark:text-slate-200 text-sm flex-1">Well Depth Over Time</h2>
        {data && (
          <span className="text-xs font-medium px-2 py-0.5 rounded-full text-white" style={{ backgroundColor: data.color }}>
            {data.aquifer}
          </span>
        )}
      </div>

      {/* Well selector */}
      <div className="px-4 sm:px-5 pt-3 sm:pt-4 pb-2">
        <div className="relative">
          <select
            className="w-full appearance-none bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-xl px-4 py-2.5 pr-9 text-sm text-slate-700 dark:text-slate-200 font-medium focus:outline-none focus:ring-2 focus:ring-sky-400 focus:border-sky-400 transition-colors cursor-pointer"
            value={selectedWell ?? ''}
            onChange={(e) => onSelectWell(e.target.value || null)}
          >
            <option value="">Choose a well to inspect…</option>
            {wellNames.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
          <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 dark:text-slate-500">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </div>
      </div>

      {/* Chart area */}
      <div className="flex-1 relative min-h-[260px] sm:min-h-[300px] px-2 pb-2">
        {!selectedWell && !loading && (
          <div className="flex flex-col items-center justify-center h-60 sm:h-72 text-slate-400 dark:text-slate-500 gap-3">
            <div className="text-4xl opacity-20">📉</div>
            <p className="text-sm text-center px-4">Select a well above or click a map marker</p>
          </div>
        )}
        {loading && (
          <div className="flex flex-col items-center justify-center h-60 sm:h-72 gap-3">
            <div className="w-8 h-8 border-2 border-sky-200 dark:border-sky-800 border-t-sky-500 rounded-full animate-spin" />
            <p className="text-sm text-slate-400 dark:text-slate-500">Loading…</p>
          </div>
        )}
        {error && (
          <div className="flex items-center justify-center h-60 sm:h-72 text-red-400 text-sm">{error}</div>
        )}
        {!loading && !error && data && (
          <Plot
            data={traces}
            layout={layout}
            config={{ responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }}
            useResizeHandler
            style={{ width: '100%', height: 310 }}
          />
        )}
      </div>
    </div>
  )
}
