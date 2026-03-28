import { useMemo } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import { useTheme } from '../ThemeContext'
import 'leaflet/dist/leaflet.css'

const TILE_LIGHT = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
const TILE_DARK = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
const TILE_ATTR =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'

export default function WellMap({ wells, selectedWell, onSelectWell }) {
  const { dark } = useTheme()

  const center = useMemo(() => {
    if (!wells.length) return [39.545146, -104.820199]
    const lats = wells.map((w) => w.latitude)
    const lons = wells.map((w) => w.longitude)
    return [
      lats.reduce((a, b) => a + b, 0) / lats.length,
      lons.reduce((a, b) => a + b, 0) / lons.length,
    ]
  }, [wells])

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 overflow-hidden flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100 dark:border-slate-700">
        <div className="w-7 h-7 rounded-lg bg-emerald-50 dark:bg-emerald-900/40 flex items-center justify-center text-base">
          📍
        </div>
        <h2 className="font-semibold text-slate-700 dark:text-slate-200 text-sm flex-1">Well Locations</h2>
        <span className="hidden sm:inline text-xs text-slate-400 dark:text-slate-500">Click a marker to select</span>
      </div>

      {/* Map */}
      <div className="h-[280px] sm:h-[400px] relative">
        {wells.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-400 dark:text-slate-500 text-sm">
            <div className="text-center space-y-2">
              <div className="text-3xl opacity-30">🗺️</div>
              <p>Loading map…</p>
            </div>
          </div>
        ) : (
          <MapContainer center={center} zoom={13} style={{ height: '100%', width: '100%' }} scrollWheelZoom>
            <TileLayer key={dark ? 'dark' : 'light'} url={dark ? TILE_DARK : TILE_LIGHT} attribution={TILE_ATTR} />
            {wells.map((well) => {
              const isSelected = selectedWell === well.well_name
              return (
                <CircleMarker
                  key={well.well_name}
                  center={[well.latitude, well.longitude]}
                  radius={isSelected ? 14 : 9}
                  pathOptions={{
                    color: isSelected ? (dark ? '#e2e8f0' : '#1e293b') : dark ? 'rgba(30,41,59,0.8)' : 'rgba(255,255,255,0.9)',
                    weight: isSelected ? 3 : 2,
                    fillColor: well.color,
                    fillOpacity: isSelected ? 1 : 0.85,
                  }}
                  eventHandlers={{ click: () => onSelectWell(well.well_name) }}
                >
                  <Popup>
                    <div className="text-sm leading-relaxed p-3 min-w-[180px]">
                      <p className="font-bold text-base mb-1">{well.well_name}</p>
                      <div className="space-y-1 opacity-75">
                        <p>
                          <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: well.color }} />
                          {well.aquifer}
                        </p>
                        <p>📍 {well.latitude.toFixed(4)}, {well.longitude.toFixed(4)}</p>
                        <p>📏 {well.latest_depth.toFixed(1)} ft depth</p>
                        <p>🗓 {well.latest_date}</p>
                      </div>
                      <button
                        className="mt-2 w-full text-center text-xs font-medium text-sky-600 bg-sky-50 hover:bg-sky-100 rounded-lg py-1.5 transition-colors"
                        onClick={() => onSelectWell(well.well_name)}
                      >
                        View time series →
                      </button>
                    </div>
                  </Popup>
                </CircleMarker>
              )
            })}
          </MapContainer>
        )}
      </div>

      {/* Legend */}
      <div className="px-4 sm:px-5 py-2.5 sm:py-3 border-t border-slate-100 dark:border-slate-700 flex flex-wrap gap-3 sm:gap-5 text-xs text-slate-500 dark:text-slate-400 bg-slate-50/50 dark:bg-slate-900/30">
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: '#e74c3c' }} />
          <span>Dawson Arkose</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: '#3498db' }} />
          <span>Denver Formation</span>
        </div>
        {selectedWell && (
          <div className="ml-auto font-medium text-slate-600 dark:text-slate-300 truncate max-w-[140px] sm:max-w-none">
            {selectedWell}
          </div>
        )}
      </div>
    </div>
  )
}
