import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { useTheme } from '../ThemeContext'
import { fetchAquiferAverages } from '../api'
import { baseLayout, chartColors } from '../plotlyTheme'

function hexToRgba(hex, alpha) {
  const v = hex.replace('#', '')
  const r = parseInt(v.slice(0, 2), 16)
  const g = parseInt(v.slice(2, 4), 16)
  const b = parseInt(v.slice(4, 6), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

export default function AquiferAverages({ aquiferFilter, dateRange }) {
  const { dark } = useTheme()
  const [aquifers, setAquifers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    fetchAquiferAverages({ aquifer: aquiferFilter, dateRange })
      .then(setAquifers)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [aquiferFilter, dateRange])

  const c = chartColors(dark)
  const traces = []
  const annotationLines = []

  for (const aq of aquifers) {
    const { aquifer, color, dates, means, upper_sem, lower_sem, trend } = aq

    if (dates.length >= 2) {
      traces.push({
        x: [...dates, ...[...dates].reverse()],
        y: [...upper_sem, ...[...lower_sem].reverse()],
        fill: 'toself',
        fillcolor: hexToRgba(color, dark ? 0.12 : 0.15),
        line: { color: 'transparent' },
        hoverinfo: 'skip',
        showlegend: false,
        type: 'scatter',
      })
    }

    traces.push({
      x: dates,
      y: means,
      type: 'scatter',
      mode: 'lines+markers',
      name: `${aquifer}`,
      line: { width: 2.5, color },
      marker: { size: 5, color, line: { color: dark ? '#1e293b' : '#fff', width: 1 } },
    })

    if (trend) {
      traces.push({
        x: trend.dates,
        y: trend.values,
        type: 'scatter',
        mode: 'lines',
        name: `${aquifer} trend`,
        line: { color, width: 1.5, dash: 'dot' },
        opacity: 0.65,
        showlegend: false,
      })

      const pLine =
        trend.p_value !== null
          ? `p = ${trend.p_value.toFixed(3)}  ·  significant? ${trend.p_value < 0.05 ? 'Yes ✓' : 'No'}`
          : 'p-value unavailable'

      annotationLines.push(
        `<b>${aquifer}</b><br>` +
        `Slope ≈ <b>${trend.slope_yearly.toFixed(2)} ft/yr</b>  ·  R² = ${trend.r_squared.toFixed(3)}<br>` +
        pLine
      )
    }
  }

  const plotAnnotation =
    annotationLines.length > 0
      ? [{
          xref: 'paper', yref: 'paper',
          x: 0.01, y: 0.99,
          xanchor: 'left', yanchor: 'top',
          align: 'left',
          font: { size: 11, family: 'Inter, sans-serif', color: c.annotationText },
          bgcolor: c.annotationBg,
          bordercolor: c.annotationBorder,
          borderwidth: 1.5,
          borderpad: 10,
          text: annotationLines.join('<br><br>'),
        }]
      : []

  const layout = {
    ...baseLayout(dark),
    xaxis: { ...baseLayout(dark).xaxis, title: { ...baseLayout(dark).xaxis.title, text: 'Date' } },
    yaxis: { ...baseLayout(dark).yaxis, title: { ...baseLayout(dark).yaxis.title, text: 'Avg Depth to Water (ft)' }, autorange: 'reversed' },
    annotations: plotAnnotation,
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 overflow-hidden">
      {/* Card header */}
      <div className="flex items-center gap-3 px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100 dark:border-slate-700">
        <div className="w-7 h-7 rounded-lg bg-blue-50 dark:bg-blue-900/40 flex items-center justify-center text-base">
          🌊
        </div>
        <h2 className="font-semibold text-slate-700 dark:text-slate-200 text-sm flex-1">Average Depth by Aquifer</h2>
        <span className="hidden sm:inline text-xs text-slate-400 dark:text-slate-500 bg-slate-50 dark:bg-slate-700 border border-slate-100 dark:border-slate-600 rounded-full px-3 py-1">
          Monthly · ±SEM band
        </span>
      </div>

      <div className="px-2 sm:px-3 py-3">
        {loading && (
          <div className="flex flex-col items-center justify-center h-60 sm:h-72 gap-3">
            <div className="w-8 h-8 border-2 border-blue-200 dark:border-blue-800 border-t-blue-500 rounded-full animate-spin" />
            <p className="text-sm text-slate-400 dark:text-slate-500">Loading aquifer data…</p>
          </div>
        )}
        {error && (
          <div className="flex items-center justify-center h-60 sm:h-72 text-red-400 text-sm">{error}</div>
        )}
        {!loading && !error && (
          <Plot
            data={traces}
            layout={layout}
            config={{ responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }}
            useResizeHandler
            style={{ width: '100%', height: 400 }}
          />
        )}
      </div>
    </div>
  )
}
