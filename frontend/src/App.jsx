import { useState, useEffect } from 'react'
import { fetchInfo, fetchWells } from './api'
import DataInfo from './components/DataInfo'
import WellMap from './components/WellMap'
import TimeSeries from './components/TimeSeries'
import AquiferAverages from './components/AquiferAverages'

export default function App() {
  const [info, setInfo] = useState(null)
  const [wells, setWells] = useState([])
  const [selectedWell, setSelectedWell] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([fetchInfo(), fetchWells()])
      .then(([infoData, wellsData]) => {
        setInfo(infoData)
        setWells(wellsData)
      })
      .catch((err) => setError(err.message))
  }, [])

  return (
    <div className="min-h-screen bg-slate-50">
      {/* ── Header ── */}
      <header className="bg-gradient-to-br from-slate-800 via-slate-800 to-slate-900 text-white shadow-xl">
        <div className="max-w-screen-xl mx-auto px-6 py-5 flex items-center gap-4">
          {/* Icon badge */}
          <div className="w-11 h-11 rounded-2xl bg-sky-500/20 border border-sky-400/20 flex items-center justify-center text-2xl shrink-0">
            💧
          </div>

          {/* Title block */}
          <div className="flex-1 min-w-0">
            <h1 className="text-lg font-bold tracking-tight leading-snug">
              Grandview Estates Water Well Monitor
            </h1>
            <p className="text-sm text-slate-400 mt-0.5">
              Rural Water Conservation District · Well Depth Analysis
            </p>
          </div>

          {/* Live indicator */}
          {info && (
            <div className="hidden sm:flex items-center gap-2 bg-white/5 border border-white/10 rounded-full px-3 py-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot shrink-0" />
              <span className="text-xs text-slate-300 whitespace-nowrap">
                {info.record_count.toLocaleString()} measurements
              </span>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-screen-xl mx-auto px-6 py-7 space-y-6">
        {/* Error banner */}
        {error && (
          <div className="flex items-start gap-3 bg-red-50 border border-red-200 text-red-700 rounded-2xl px-5 py-4 text-sm">
            <span className="text-base mt-0.5">⚠️</span>
            <div>
              <p className="font-semibold">Failed to load data</p>
              <p className="text-red-500 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* Stat tiles */}
        <DataInfo info={info} wells={wells} />

        {/* Map + Time Series */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <WellMap
            wells={wells}
            selectedWell={selectedWell}
            onSelectWell={setSelectedWell}
          />
          <TimeSeries
            wells={wells}
            selectedWell={selectedWell}
            onSelectWell={setSelectedWell}
          />
        </div>

        {/* Aquifer averages */}
        <AquiferAverages />

        <p className="text-center text-xs text-slate-400 pb-4">
          Grandview Estates RWCD · Data loaded from static CSV · Not a live feed
        </p>
      </main>
    </div>
  )
}
