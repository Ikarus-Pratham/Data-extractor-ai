type Props = {
  index: number
  start: string
  end: string
  onChange: (field: 'start' | 'end', value: string) => void
  onRemove: () => void
}

export default function RangeRow({ index, start, end, onChange, onRemove }: Props) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-6 gap-3 items-end">
      <div className="sm:col-span-1">
        <label className="block text-xs font-medium text-neutral-400">Product Number</label>
        <input
          type="text"
          value={String(index + 1)}
          disabled
          className="mt-1 w-full rounded-lg border border-neutral-800 bg-neutral-800 px-3 py-2 text-sm text-neutral-300"
        />
      </div>
      <div className="sm:col-span-2">
        <label className="block text-xs font-medium text-neutral-400">First page number</label>
        <input
          type="number"
          min={1}
          value={start}
          onChange={(e) => onChange('start', e.target.value)}
          placeholder="e.g. 1"
          className="mt-1 w-full rounded-lg border border-neutral-800 bg-neutral-800 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-500 focus:border-blue-500 focus:ring-blue-500"
        />
      </div>
      <div className="sm:col-span-2">
        <label className="block text-xs font-medium text-neutral-400">Last page number</label>
        <input
          type="number"
          min={1}
          value={end}
          onChange={(e) => onChange('end', e.target.value)}
          placeholder="e.g. 2"
          className="mt-1 w-full rounded-lg border border-neutral-800 bg-neutral-800 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-500 focus:border-blue-500 focus:ring-blue-500"
        />
      </div>
      <div className="sm:col-span-1">
        <button
          type="button"
          onClick={onRemove}
          className="w-full rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm text-neutral-200 hover:bg-neutral-800"
        >
          Remove
        </button>
      </div>
    </div>
  )
}


