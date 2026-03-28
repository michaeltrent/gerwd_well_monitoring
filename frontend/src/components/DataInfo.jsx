const AQUIFER_COLORS = {
  'Dawson Arkose': '#e74c3c',
  'Denver Formation': '#3498db',
}

function StatTile({ icon, label, children, loading }) {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-5">
      <div className="flex items-center gap-2.5 mb-3">
        <div className="w-8 h-8 rounded-xl bg-sky-50 flex items-center justify-center text-sky-500 text-base shrink-0">
          {icon}
        </div>
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
          {label}
        </span>
      </div>
      {loading ? (
        <div className="h-6 w-24 bg-slate-100 rounded animate-pulse" />
      ) : (
        <div className="text-slate-800">{children}</div>
      )}
    </div>
  )
}

export default function DataInfo({ info, wells }) {
  const loading = !info

  const wellCount = wells?.length ?? 0
  const aquiferCount = info?.aquifers?.length ?? 0

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
      {/* Wells */}
      <StatTile icon="🔵" label="Wells" loading={loading}>
        <p className="text-3xl font-bold tracking-tight">{wellCount}</p>
        <p className="text-xs text-slate-400 mt-0.5">monitoring locations</p>
      </StatTile>

      {/* Measurements */}
      <StatTile icon="📊" label="Measurements" loading={loading}>
        <p className="text-3xl font-bold tracking-tight">
          {info?.record_count.toLocaleString() ?? '—'}
        </p>
        <p className="text-xs text-slate-400 mt-0.5">depth readings</p>
      </StatTile>

      {/* Date range */}
      <StatTile icon="📅" label="Date Range" loading={loading}>
        {info?.date_range ? (
          <>
            <p className="text-sm font-semibold leading-snug">
              {info.date_range.min}
            </p>
            <p className="text-xs text-slate-400 mt-0.5">→ {info.date_range.max}</p>
          </>
        ) : (
          <p className="text-sm text-slate-400">N/A</p>
        )}
      </StatTile>

      {/* Aquifers */}
      <StatTile icon="🌊" label="Aquifers" loading={loading}>
        <div className="flex flex-col gap-1.5 mt-0.5">
          {info?.aquifers.map((aq) => (
            <span
              key={aq}
              className="inline-flex items-center px-2 py-0.5 rounded-full text-white text-xs font-medium w-fit"
              style={{ backgroundColor: AQUIFER_COLORS[aq] ?? '#64748b' }}
            >
              {aq}
            </span>
          ))}
        </div>
      </StatTile>
    </div>
  )
}
