import type { ChangeEvent } from 'react'

type Props = {
  file: File | null
  onFileChange: (file: File | null) => void
}

export default function FileUpload({ file, onFileChange }: Props) {
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    onFileChange(e.target.files?.[0] ?? null)
  }

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-neutral-200">Upload PDF</label>
      <input
        type="file"
        accept="application/pdf"
        onChange={handleChange}
        className="block w-full text-sm file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-neutral-800 file:text-neutral-200 hover:file:bg-neutral-700/90 focus:outline-none"
      />
      {file && (
        <p className="text-xs text-neutral-400">Selected: {file.name}</p>
      )}
    </div>
  )
}


