import { useTheme } from '../ThemeContext'

const AQUIFER_OPTIONS = [
  { label: 'All', value: null },
  { label: 'Dawson Arkose', value: 'Dawson Arkose', color: '#e74c3c' },
  { label: 'Denver Formation', value: 'Denver Formation', color: '#3498db' },
]

export default function FilterBar({ aquiferFilter, onAquiferChange, dateRange, onDateChange, dateBounds }) {
  const { dark } = useTheme()

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 px-4 sm:px-5 py-3 sm:py-4">
      <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-5">
        {/* Aquifer toggle */}
        <div className="flex items-center gap-2.5 flex-wrap">
          <span className="text-[10px] sm:text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-widest shrink-0">
            Aquifer
          </span>
          <div className="flex gap-0.5 bg-slate-100 dark:bg-slate-700/60 rounded-xl p-0.5">
            {AQUIFER_OPTIONS.map((opt) => {
              const active = aquiferFilter === opt.value
              return (
                <button
                  key={opt.label}
                  onClick={() => onAquiferChange(opt.value)}
                  className={[
                    'rounded-lg px-2.5 sm:px-3 py-1.5 text-xs font-medium transition-all whitespace-nowrap',
                    active
                      ? 'bg-white dark:bg-slate-600 shadow-sm text-slate-800 dark:text-slate-100'
                      : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200',
                  ].join(' ')}
                >
                  {opt.color && (
                    <span
                      className="inline-block w-2 h-2 rounded-full mr-1.5 align-middle"
                      style={{ backgroundColor: opt.color }}
                    />
                  )}
                  {opt.label}
                </button>
              )
            })}
          </div>
        </div>

        {/* Date range */}
        <div className="flex items-center gap-2 sm:ml-auto flex-wrap">
          <span className="text-[10px] sm:text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-widest shrink-0">
            Dates
          </span>
          <input
            type="date"
            value={dateRange.start ?? ''}
            min={dateBounds?.min}
            max={dateRange.end ?? dateBounds?.max}
            onChange={(e) => onDateChange({ ...dateRange, start: e.target.value || null })}
            className="bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400 transition-colors w-[130px]"
          />
          <span className="text-slate-300 dark:text-slate-600 text-xs">→</span>
          <input
            type="date"
            value={dateRange.end ?? ''}
            min={dateRange.start ?? dateBounds?.min}
            max={dateBounds?.max}
            onChange={(e) => onDateChange({ ...dateRange, end: e.target.value || null })}
            className="bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400 transition-colors w-[130px]"
          />
          {(dateRange.start || dateRange.end) && (
            <button
              onClick={() => onDateChange({ start: null, end: null })}
              className="w-7 h-7 flex items-center justify-center rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors text-xs"
              aria-label="Clear dates"
              title="Clear date filter"
            >
              ✕
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
