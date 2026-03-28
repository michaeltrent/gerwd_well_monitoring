import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { fetchAquiferAverages } from '../api'

function hexToRgba(hex, alpha) {
  const v = hex.replace('#', '')
  const r = parseInt(v.slice(0, 2), 16)
  const g = parseInt(v.slice(2, 4), 16)
  const b = parseInt(v.slice(4, 6), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

export default function AquiferAverages() {
  const [aquifers, setAquifers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchAquiferAverages()
      .then(setAquifers)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  const traces = []
  const annotationLines = []

  for (const aq of aquifers) {
    const { aquifer, color, dates, means, upper_sem, lower_sem, trend } = aq

    // ±SEM shaded band
    if (dates.length >= 2) {
      traces.push({
        x: [...dates, ...[...dates].reverse()],
        y: [...upper_sem, ...[...lower_sem].reverse()],
        fill: 'toself',
        fillcolor: hexToRgba(color, 0.15),
        line: { color: 'transparent' },
        hoverinfo: 'skip',
        showlegend: false,
        type: 'scatter',
      })
    }

    // Mean line
    traces.push({
      x: dates,
      y: means,
      type: 'scatter',
      mode: 'lines+markers',
      name: `${aquifer}`,
      line: { width: 2.5, color },
      marker: { size: 5, color, line: { color: '#fff', width: 1 } },
    })

    // Trend line
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
          font: { size: 11, family: 'Inter, sans-serif', color: '#475569' },
          bgcolor: 'rgba(255,255,255,0.92)',
          bordercolor: '#cbd5e1',
          borderwidth: 1.5,
          borderpad: 10,
          text: annotationLines.join('<br><br>'),
        }]
      : []

  const layout = {
    font: { family: 'Inter, sans-serif' },
    margin: { l: 52, r: 16, t: 20, b: 48 },
    hovermode: 'x unified',
    plot_bgcolor: '#f8fafc',
    paper_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {
      title: { text: 'Date', font: { size: 11, color: '#94a3b8' } },
      gridcolor: '#e2e8f0',
      linecolor: '#cbd5e1',
      tickfont: { size: 10, color: '#94a3b8' },
    },
    yaxis: {
      title: { text: 'Avg Depth to Water (ft)', font: { size: 11, color: '#94a3b8' } },
      autorange: 'reversed',
      gridcolor: '#e2e8f0',
      linecolor: '#cbd5e1',
      tickfont: { size: 10, color: '#94a3b8' },
    },
    legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1, font: { size: 11 } },
    annotations: plotAnnotation,
    autosize: true,
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
      {/* Card header */}
      <div className="flex items-center gap-3 px-5 py-4 border-b border-slate-100">
        <div className="w-7 h-7 rounded-lg bg-blue-50 flex items-center justify-center text-base">
          🌊
        </div>
        <h2 className="font-semibold text-slate-700 text-sm flex-1">Average Depth by Aquifer</h2>
        <span className="text-xs text-slate-400 bg-slate-50 border border-slate-100 rounded-full px-3 py-1">
          Monthly · ±SEM band
        </span>
      </div>

      <div className="px-3 py-3">
        {loading && (
          <div className="flex flex-col items-center justify-center h-72 gap-3">
            <div className="w-8 h-8 border-2 border-blue-200 border-t-blue-500 rounded-full animate-spin" />
            <p className="text-sm text-slate-400">Loading aquifer data…</p>
          </div>
        )}
        {error && (
          <div className="flex items-center justify-center h-72 text-red-400 text-sm">{error}</div>
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
