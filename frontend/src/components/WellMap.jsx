import { useMemo } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

function CardHeader({ children }) {
  return (
    <div className="flex items-center gap-3 px-5 py-4 border-b border-slate-100">
      <div className="w-7 h-7 rounded-lg bg-emerald-50 flex items-center justify-center text-base">
        📍
      </div>
      {children}
    </div>
  )
}

export default function WellMap({ wells, selectedWell, onSelectWell }) {
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
    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden flex flex-col">
      <CardHeader>
        <h2 className="font-semibold text-slate-700 text-sm flex-1">Well Locations</h2>
        <span className="text-xs text-slate-400">Click a marker to select</span>
      </CardHeader>

      <div style={{ height: 400 }} className="relative">
        {wells.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-400 text-sm">
            <div className="text-center space-y-2">
              <div className="text-3xl opacity-30">🗺️</div>
              <p>Loading map…</p>
            </div>
          </div>
        ) : (
          <MapContainer
            center={center}
            zoom={13}
            style={{ height: '100%', width: '100%' }}
            scrollWheelZoom
          >
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            />
            {wells.map((well) => {
              const isSelected = selectedWell === well.well_name
              return (
                <CircleMarker
                  key={well.well_name}
                  center={[well.latitude, well.longitude]}
                  radius={isSelected ? 14 : 9}
                  pathOptions={{
                    color: isSelected ? '#1e293b' : 'rgba(255,255,255,0.9)',
                    weight: isSelected ? 3 : 2,
                    fillColor: well.color,
                    fillOpacity: isSelected ? 1 : 0.85,
                  }}
                  eventHandlers={{ click: () => onSelectWell(well.well_name) }}
                >
                  <Popup>
                    <div className="text-sm leading-relaxed p-3 min-w-[180px]">
                      <p className="font-bold text-slate-800 text-base mb-1">{well.well_name}</p>
                      <div className="space-y-1 text-slate-500">
                        <p>
                          <span
                            className="inline-block w-2 h-2 rounded-full mr-1.5"
                            style={{ backgroundColor: well.color }}
                          />
                          {well.aquifer}
                        </p>
                        <p>📍 {well.latitude.toFixed(4)}, {well.longitude.toFixed(4)}</p>
                        <p>📏 {well.latest_depth.toFixed(1)} ft depth</p>
                        <p>🗓 {well.latest_date}</p>
                      </div>
                      <button
                        className="mt-2 w-full text-center text-xs font-medium text-sky-600 hover:text-sky-700 bg-sky-50 hover:bg-sky-100 rounded-lg py-1.5 transition-colors"
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
      <div className="px-5 py-3 border-t border-slate-100 flex gap-5 text-xs text-slate-500 bg-slate-50/50">
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: '#e74c3c' }} />
          <span>Dawson Arkose</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: '#3498db' }} />
          <span>Denver Formation</span>
        </div>
        {selectedWell && (
          <div className="ml-auto font-medium text-slate-600">
            Selected: {selectedWell}
          </div>
        )}
      </div>
    </div>
  )
}
