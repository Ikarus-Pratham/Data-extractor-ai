import './App.css'
import { useState } from 'react'
import FileUpload from './components/FileUpload.tsx'
import RangeList, { type PageRange } from './components/RangeList.tsx'

function App() {
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [ranges, setRanges] = useState<PageRange[]>([])
  const [message, setMessage] = useState<string | null>(null)

  const backendUrl = (import.meta as any).env?.VITE_BACKEND_URL || 'https://smhxghmg-1234.inc1.devtunnels.ms/ai/api/extract'

  const handleAddRange = () => {
    setRanges((prev) => [...prev, { start: '', end: '' }])
  }

  const handleChangeRange = (index: number, field: keyof PageRange, value: string) => {
    setRanges((prev) => {
      const next = [...prev]
      next[index] = { ...next[index], [field]: value }
      return next
    })
  }

  const handleRemoveRange = (index: number) => {
    setRanges((prev) => prev.filter((_, i) => i !== index))
  }

  const buildPagesPayload = () => {
    const pages: Record<string, [number, number]> = {}
    ranges.forEach((range, idx) => {
      const start = Number(range.start)
      const end = Number(range.end)
      if (!Number.isFinite(start) || !Number.isFinite(end)) return
      if (start <= 0 || end <= 0) return
      if (end < start) return
      const key = String(idx + 1)
      pages[key] = [start, end]
    })
    return pages
  }

  const handleSubmit = async () => {
    setMessage(null)
    if (!pdfFile) {
      setMessage('Please select a PDF file.')
      return
    }
    if (ranges.length === 0) {
      setMessage('Please add at least one page range.')
      return
    }

    const pages = buildPagesPayload()
    if (Object.keys(pages).length === 0) {
      setMessage('Please provide valid page ranges (positive numbers, end >= start).')
      return
    }

    const formData = new FormData()
    formData.append('pdf_file', pdfFile)
    formData.append('pages', JSON.stringify(pages))
    // Fire-and-forget: do not await server response
    try {
      void fetch(backendUrl, {
        method: 'POST',
        body: formData,
      }).catch(() => {
        // Intentionally ignore async errors; frontend does not wait for response
      })
      setMessage('Request sent.')
    } catch (err: any) {
      // Only handles synchronous errors (rare for fetch init)
      setMessage(err?.message || 'Failed to initiate request.')
    }
  }

  return (
    <>
      <div className="min-h-screen bg-neutral-950 text-neutral-100">
        <header className="border-b border-neutral-800/80 bg-neutral-950/60 backdrop-blur supports-[backdrop-filter]:bg-neutral-950/40">
          <div className="max-w-4xl mx-auto px-4 py-4">
            <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">PDF Page Extractor</h1>
          </div>
        </header>
        <main className="max-w-4xl mx-auto px-4 py-8 space-y-6">
          <section className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.02)_inset]">
            <FileUpload file={pdfFile} onFileChange={setPdfFile} />
          </section>

          <section className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.02)_inset]">
            <RangeList
              ranges={ranges}
              onAddRange={handleAddRange}
              onChangeRange={handleChangeRange}
              onRemoveRange={handleRemoveRange}
            />
          </section>

          <section className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleSubmit}
              className="inline-flex items-center rounded-lg bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white shadow hover:bg-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/60 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              Send
            </button>
            {message && <p className="text-sm text-neutral-300">{message}</p>}
          </section>

          <p className="text-xs text-neutral-500">
            Sends multipart/form-data: <code>pdf_file</code> (File) and <code>pages</code> (JSON).
          </p>
        </main>
      </div>
    </>
  )
}

export default App
