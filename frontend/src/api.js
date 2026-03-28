/**
 * Static-build API layer — all data processing happens client-side.
 * Exports the same function signatures the components expect.
 */
import { loadData, getInfo, getWells, getTimeseries, getAquiferAverages } from './dataUtils'

let _rows = null
async function ensureData() {
  if (!_rows) _rows = await loadData()
  return _rows
}

export async function fetchInfo() {
  return getInfo(await ensureData())
}

export async function fetchWells() {
  return getWells(await ensureData())
}

export async function fetchTimeseries(well, dateRange) {
  const data = getTimeseries(await ensureData(), well, dateRange)
  if (!data) throw new Error(`No data for '${well}' in this date range`)
  return data
}

export async function fetchAquiferAverages({ aquifer, dateRange } = {}) {
  return getAquiferAverages(await ensureData(), { aquifer, dateRange })
}
