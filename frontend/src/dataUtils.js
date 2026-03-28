/**
 * Client-side replacement for backend/main.py.
 * Parses the CSV, cleans data, computes regression — all in the browser.
 */
import Papa from 'papaparse'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const CSV_PATH = import.meta.env.BASE_URL + 'well_data_df_10.07.2025.csv'

const AQUIFER_COLORS = {
  'Dawson Arkose': '#e74c3c',
  'Denver Formation': '#3498db',
}

const COLUMN_ALIASES = {
  well_name: 'Well_Name',
  well: 'Well_Name',
  local_aquafer: 'Aquifer',
  local_aquifer: 'Aquifer',
  aquifer: 'Aquifer',
  phenomenontime: 'Date',
  date: 'Date',
  result: 'Depth_ft',
  depth_ft: 'Depth_ft',
  'water level': 'Depth_ft',
  resultmethod: 'Method',
  method: 'Method',
  latitude: 'Latitude',
  lat: 'Latitude',
  longitude: 'Longitude',
  lon: 'Longitude',
  long: 'Longitude',
}

const ESSENTIAL = ['Well_Name', 'Latitude', 'Longitude', 'Aquifer', 'Date', 'Depth_ft']

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------
function erfc(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const poly =
    t *
    (0.254829592 +
      t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
  const result = poly * Math.exp(-x * x)
  return x >= 0 ? result : 2 - result
}

function normalApproxPValue(tStat) {
  return 2 * 0.5 * erfc(Math.abs(tStat) / Math.sqrt(2))
}

function linearRegression(x, y) {
  const n = x.length
  if (n < 2) return null

  const meanX = x.reduce((a, b) => a + b, 0) / n
  const meanY = y.reduce((a, b) => a + b, 0) / n

  let ssx = 0, ssxy = 0
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX
    ssx += dx * dx
    ssxy += dx * (y[i] - meanY)
  }
  if (ssx === 0) return null

  const slope = ssxy / ssx
  const intercept = meanY - slope * meanX
  const yHat = x.map((xi) => slope * xi + intercept)

  let ssRes = 0, ssTot = 0
  for (let i = 0; i < n; i++) {
    ssRes += (y[i] - yHat[i]) ** 2
    ssTot += (y[i] - meanY) ** 2
  }
  const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0

  let pValue = null
  if (n > 2 && ssx > 0) {
    const sErr = Math.sqrt(ssRes / (n - 2))
    const slopeStderr = sErr / Math.sqrt(ssx)
    if (slopeStderr > 0) {
      const tStat = slope / slopeStderr
      pValue = normalApproxPValue(tStat)
    }
  }

  return { slope, intercept, rSquared, pValue, yHat }
}

// ---------------------------------------------------------------------------
// Data loading & cleaning
// ---------------------------------------------------------------------------
function standardizeColumns(row) {
  const out = {}
  for (const [key, val] of Object.entries(row)) {
    const trimmed = key.trim()
    if (trimmed.toLowerCase().startsWith('unnamed')) continue
    const target = COLUMN_ALIASES[trimmed.toLowerCase()]
    out[target ?? trimmed] = val
  }
  if (!('Method' in out)) out.Method = 'Unknown'
  return out
}

function coerceRow(row) {
  const r = standardizeColumns(row)
  for (const col of ESSENTIAL) {
    if (r[col] == null || r[col] === '') return null
  }
  r.Latitude = parseFloat(r.Latitude)
  r.Longitude = parseFloat(r.Longitude)
  r.Depth_ft = parseFloat(r.Depth_ft)
  r.Date = new Date(r.Date)

  if ([r.Latitude, r.Longitude, r.Depth_ft].some(Number.isNaN)) return null
  if (Number.isNaN(r.Date.getTime())) return null

  r.Well_Name = String(r.Well_Name).trim()
  r.Aquifer = String(r.Aquifer).trim()
  const m = String(r.Method ?? '').trim()
  r.Method = !m || m === 'nan' || m === 'None' ? 'Unknown' : m

  return r
}

let _cache = null

export async function loadData() {
  if (_cache) return _cache

  const res = await fetch(CSV_PATH)
  const text = await res.text()
  const { data } = Papa.parse(text, { header: true, skipEmptyLines: true })

  const rows = data.map(coerceRow).filter(Boolean)
  rows.sort((a, b) => {
    const c = a.Well_Name.localeCompare(b.Well_Name)
    return c !== 0 ? c : a.Date - b.Date
  })

  _cache = rows
  return rows
}

// ---------------------------------------------------------------------------
// Query functions
// ---------------------------------------------------------------------------
const fmtDate = (d) => d.toISOString().slice(0, 10)

function applyDateFilter(rows, dateRange) {
  if (!dateRange) return rows
  let out = rows
  if (dateRange.start) {
    const s = new Date(dateRange.start)
    out = out.filter((r) => r.Date >= s)
  }
  if (dateRange.end) {
    const e = new Date(dateRange.end)
    e.setDate(e.getDate() + 1) // make end date inclusive
    out = out.filter((r) => r.Date < e)
  }
  return out
}

export function getInfo(rows) {
  if (!rows.length)
    return { filename: 'well_data_df_10.07.2025.csv', record_count: 0, aquifers: [], date_range: null }

  const aquifers = [...new Set(rows.map((r) => r.Aquifer))].sort()
  const dates = rows.map((r) => r.Date)
  const min = new Date(Math.min(...dates))
  const max = new Date(Math.max(...dates))

  return {
    filename: 'well_data_df_10.07.2025.csv',
    record_count: rows.length,
    aquifers,
    date_range: { min: fmtDate(min), max: fmtDate(max) },
  }
}

export function getWells(rows) {
  if (!rows.length) return []
  const latest = new Map()
  for (const r of rows) {
    const prev = latest.get(r.Well_Name)
    if (!prev || r.Date > prev.Date) latest.set(r.Well_Name, r)
  }
  return [...latest.values()].map((r) => ({
    well_name: r.Well_Name,
    latitude: r.Latitude,
    longitude: r.Longitude,
    aquifer: r.Aquifer,
    latest_depth: r.Depth_ft,
    latest_date: fmtDate(r.Date),
    color: AQUIFER_COLORS[r.Aquifer] ?? '#666666',
  }))
}

export function getTimeseries(rows, well, dateRange) {
  let sdf = rows.filter((r) => r.Well_Name === well)
  sdf = applyDateFilter(sdf, dateRange)
  sdf.sort((a, b) => a.Date - b.Date)
  if (!sdf.length) return null

  const aquifer = sdf[sdf.length - 1].Aquifer
  const color = AQUIFER_COLORS[aquifer] ?? '#666666'
  const dates = sdf.map((r) => fmtDate(r.Date))
  const depths = sdf.map((r) => r.Depth_ft)

  const baseDate = sdf[0].Date
  const xDays = sdf.map((r) => (r.Date - baseDate) / 86400000)
  const reg = linearRegression(xDays, depths)

  let trend = null
  if (reg) {
    trend = {
      dates,
      values: reg.yHat,
      slope_daily: reg.slope,
      slope_yearly: reg.slope * 365.25,
      intercept: reg.intercept,
      r_squared: reg.rSquared,
      p_value: reg.pValue,
      base_date: fmtDate(baseDate),
    }
  }

  return { well, aquifer, color, dates, depths, trend }
}

export function getAquiferAverages(rows, { aquifer: aquiferFilter, dateRange } = {}) {
  let filtered = rows
  if (aquiferFilter) filtered = filtered.filter((r) => r.Aquifer === aquiferFilter)
  filtered = applyDateFilter(filtered, dateRange)
  if (!filtered.length) return []

  // Group by (aquifer, month)
  const buckets = new Map()
  for (const r of filtered) {
    const monthKey = `${r.Aquifer}|${r.Date.getFullYear()}-${String(r.Date.getMonth() + 1).padStart(2, '0')}`
    if (!buckets.has(monthKey)) buckets.set(monthKey, { aquifer: r.Aquifer, year: r.Date.getFullYear(), month: r.Date.getMonth(), values: [] })
    buckets.get(monthKey).values.push(r.Depth_ft)
  }

  // Compute stats per bucket
  const byAquifer = new Map()
  for (const b of buckets.values()) {
    if (!byAquifer.has(b.aquifer)) byAquifer.set(b.aquifer, [])
    const vals = b.values
    const n = vals.length
    const mean = vals.reduce((a, v) => a + v, 0) / n
    const std = n > 1 ? Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / (n - 1)) : 0
    const sem = n > 0 ? std / Math.sqrt(n) : 0
    byAquifer.get(b.aquifer).push({
      date: new Date(b.year, b.month, 1),
      mean,
      sem,
    })
  }

  const result = []
  for (const [aquifer, entries] of byAquifer) {
    entries.sort((a, b) => a.date - b.date)
    const color = AQUIFER_COLORS[aquifer] ?? '#666666'
    const dates = entries.map((e) => fmtDate(e.date))
    const means = entries.map((e) => e.mean)
    const upperSem = entries.map((e) => e.mean + e.sem)
    const lowerSem = entries.map((e) => e.mean - e.sem)

    let trend = null
    if (entries.length >= 2) {
      const baseDate = entries[0].date
      const xDays = entries.map((e) => (e.date - baseDate) / 86400000)
      const reg = linearRegression(xDays, means)
      if (reg) {
        trend = {
          dates,
          values: reg.yHat,
          slope_daily: reg.slope,
          slope_yearly: reg.slope * 365.25,
          intercept: reg.intercept,
          r_squared: reg.rSquared,
          p_value: reg.pValue,
          base_date: fmtDate(baseDate),
        }
      }
    }

    result.push({ aquifer, color, dates, means, upper_sem: upperSem, lower_sem: lowerSem, trend })
  }

  return result
}
