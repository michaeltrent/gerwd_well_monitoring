/**
 * Static-build API layer — all data processing happens client-side.
 * Exports the same function signatures as the original fetch-based API
 * so no component changes are needed.
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

export async function fetchTimeseries(well) {
  const data = getTimeseries(await ensureData(), well)
  if (!data) throw new Error(`Well '${well}' not found`)
  return data
}

export async function fetchAquiferAverages() {
  return getAquiferAverages(await ensureData())
}
