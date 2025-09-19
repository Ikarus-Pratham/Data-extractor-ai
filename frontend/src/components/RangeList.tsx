import RangeRow from './RangeRow'

export type PageRange = { start: string; end: string }

type Props = {
  ranges: PageRange[]
  onAddRange: () => void
  onChangeRange: (index: number, field: keyof PageRange, value: string) => void
  onRemoveRange: (index: number) => void
}

export default function RangeList({ ranges, onAddRange, onChangeRange, onRemoveRange }: Props) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-neutral-100">Page Ranges</h2>
        <button
          type="button"
          onClick={onAddRange}
          className="inline-flex items-center rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/60"
        >
          Add Range
        </button>
      </div>

      {ranges.length === 0 && (
        <p className="text-sm text-neutral-400">No ranges added. Click "Add Range" to begin.</p>
      )}

      <div className="space-y-3">
        {ranges.map((range, idx) => (
          <RangeRow
            key={idx}
            index={idx}
            start={range.start}
            end={range.end}
            onChange={(field, value) => onChangeRange(idx, field, value)}
            onRemove={() => onRemoveRange(idx)}
          />
        ))}
      </div>
    </div>
  )
}


