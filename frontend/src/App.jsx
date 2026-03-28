import { useState, useEffect, useMemo } from 'react'
import { useTheme } from './ThemeContext'
import { fetchInfo, fetchWells } from './api'
import DataInfo from './components/DataInfo'
import FilterBar from './components/FilterBar'
import WellMap from './components/WellMap'
import TimeSeries from './components/TimeSeries'
import AquiferAverages from './components/AquiferAverages'

function ThemeToggle() {
  const { dark, toggle } = useTheme()
  return (
    <button
      onClick={toggle}
      aria-label="Toggle dark mode"
      className="w-9 h-9 rounded-xl bg-white/10 hover:bg-white/20 border border-white/10 flex items-center justify-center transition-colors shrink-0"
    >
      {dark ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="5" />
          <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  )
}

export default function App() {
  const [info, setInfo] = useState(null)
  const [wells, setWells] = useState([])
  const [selectedWell, setSelectedWell] = useState(null)
  const [error, setError] = useState(null)

  // Filters
  const [aquiferFilter, setAquiferFilter] = useState(null)
  const [dateRange, setDateRange] = useState({ start: null, end: null })

  useEffect(() => {
    Promise.all([fetchInfo(), fetchWells()])
      .then(([infoData, wellsData]) => {
        setInfo(infoData)
        setWells(wellsData)
      })
      .catch((err) => setError(err.message))
  }, [])

  // Filter wells for map + dropdown
  const filteredWells = useMemo(() => {
    if (!aquiferFilter) return wells
    return wells.filter((w) => w.aquifer === aquiferFilter)
  }, [wells, aquiferFilter])

  // If current selection is no longer visible, clear it
  useEffect(() => {
    if (selectedWell && filteredWells.length && !filteredWells.some((w) => w.well_name === selectedWell)) {
      setSelectedWell(null)
    }
  }, [filteredWells, selectedWell])

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 transition-colors">
      {/* ── Header ── */}
      <header className="bg-gradient-to-br from-slate-800 via-slate-800 to-slate-900 dark:from-slate-900 dark:via-slate-900 dark:to-black text-white shadow-xl">
        <div className="max-w-screen-xl mx-auto px-4 sm:px-6 py-4 sm:py-5 flex items-center gap-3 sm:gap-4">
          <div className="w-10 h-10 sm:w-11 sm:h-11 rounded-2xl bg-sky-500/20 border border-sky-400/20 flex items-center justify-center text-xl sm:text-2xl shrink-0">
            💧
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="text-base sm:text-lg font-bold tracking-tight leading-snug truncate">
              Grandview Estates Well Monitor
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 mt-0.5 truncate">
              Rural Water Conservation District
            </p>
          </div>
          {info && (
            <div className="hidden md:flex items-center gap-2 bg-white/5 border border-white/10 rounded-full px-3 py-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot shrink-0" />
              <span className="text-xs text-slate-300 whitespace-nowrap">
                {info.record_count.toLocaleString()} measurements
              </span>
            </div>
          )}
          <ThemeToggle />
        </div>
      </header>

      <main className="max-w-screen-xl mx-auto px-4 sm:px-6 py-5 sm:py-7 space-y-5 sm:space-y-6">
        {error && (
          <div className="flex items-start gap-3 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 rounded-2xl px-4 sm:px-5 py-4 text-sm">
            <span className="text-base mt-0.5">⚠️</span>
            <div>
              <p className="font-semibold">Failed to load data</p>
              <p className="text-red-500 dark:text-red-400 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* Stat tiles */}
        <DataInfo info={info} wells={wells} />

        {/* Filters */}
        <FilterBar
          aquiferFilter={aquiferFilter}
          onAquiferChange={setAquiferFilter}
          dateRange={dateRange}
          onDateChange={setDateRange}
          dateBounds={info?.date_range}
        />

        {/* Map + Time Series */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-6">
          <WellMap
            wells={filteredWells}
            selectedWell={selectedWell}
            onSelectWell={setSelectedWell}
          />
          <TimeSeries
            wells={filteredWells}
            selectedWell={selectedWell}
            onSelectWell={setSelectedWell}
            dateRange={dateRange}
          />
        </div>

        {/* Aquifer averages */}
        <AquiferAverages aquiferFilter={aquiferFilter} dateRange={dateRange} />

        <p className="text-center text-xs text-slate-400 dark:text-slate-600 pb-4">
          Grandview Estates RWCD · Data loaded from static CSV · Not a live feed
        </p>
      </main>
    </div>
  )
}
