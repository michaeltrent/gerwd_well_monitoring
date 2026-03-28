/** Shared Plotly layout colors for light/dark mode. */
export function chartColors(dark) {
  if (dark) {
    return {
      plotBg: '#0f172a',
      paperBg: 'rgba(0,0,0,0)',
      gridColor: '#1e293b',
      lineColor: '#334155',
      tickColor: '#64748b',
      titleColor: '#94a3b8',
      legendColor: '#cbd5e1',
      annotationBg: 'rgba(15,23,42,0.92)',
      annotationText: '#cbd5e1',
      annotationBorder: '#475569',
    }
  }
  return {
    plotBg: '#f8fafc',
    paperBg: 'rgba(0,0,0,0)',
    gridColor: '#e2e8f0',
    lineColor: '#cbd5e1',
    tickColor: '#94a3b8',
    titleColor: '#94a3b8',
    legendColor: '#475569',
    annotationBg: 'rgba(255,255,255,0.92)',
    annotationText: '#475569',
    annotationBorder: '#cbd5e1',
  }
}

export function baseLayout(dark) {
  const c = chartColors(dark)
  return {
    font: { family: 'Inter, sans-serif', color: c.legendColor },
    margin: { l: 52, r: 16, t: 20, b: 48 },
    hovermode: 'x unified',
    plot_bgcolor: c.plotBg,
    paper_bgcolor: c.paperBg,
    xaxis: {
      title: { font: { size: 11, color: c.titleColor } },
      gridcolor: c.gridColor,
      linecolor: c.lineColor,
      tickfont: { size: 10, color: c.tickColor },
    },
    yaxis: {
      title: { font: { size: 11, color: c.titleColor } },
      gridcolor: c.gridColor,
      linecolor: c.lineColor,
      tickfont: { size: 10, color: c.tickColor },
    },
    legend: {
      orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1,
      font: { size: 11, color: c.legendColor },
    },
    autosize: true,
  }
}
