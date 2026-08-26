import { useState, useEffect, useCallback, useRef, type DragEvent, type ChangeEvent } from 'react'
import { Button } from '@/components/ui/button'
import { uploadPDF, getUploadProgress } from '@/services/api'
import { useChatStore } from '@/store/chatStore'
import { Upload, FileText, Loader2, CheckCircle2, XCircle } from 'lucide-react'
import { toast } from 'sonner'

export function FileUpload() {
  const { uploadStatus, setUploadStatus, lastUploadFilename } = useChatStore()
  const [isDragOver, setIsDragOver] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [progress, setProgress] = useState({ step: '', pct: 0 })
  const progressFailures = useRef(0)

  // 轮询入库进度
  useEffect(() => {
    if (!taskId || uploadStatus !== 'uploading') return
    progressFailures.current = 0
    let active = true
    const checkProgress = async () => {
      try {
        const p = await getUploadProgress(taskId)
        if (!active) return
        progressFailures.current = 0
        setProgress({ step: p.step, pct: p.progress_pct })
        if (p.status === 'success') {
          setUploadStatus('success', p.filename)
          toast.success('入库完成！', { description: '文件已成功解析入库，现在可以提问了。' })
          setTaskId(null)
        } else if (p.status === 'error') {
          setUploadStatus('error')
          toast.error('入库失败', { description: p.error || '未知错误' })
          setTaskId(null)
        }
      } catch {
        // 网络短暂中断时继续轮询；连续失败避免界面永久停留在转圈状态。
        progressFailures.current += 1
        if (progressFailures.current >= 20 && active) {
          setUploadStatus('error')
          toast.error('无法获取入库进度', { description: '请检查后端服务后重试。' })
          setTaskId(null)
        }
      }
    }
    void checkProgress()
    const interval = setInterval(() => void checkProgress(), 1500)
    return () => {
      active = false
      clearInterval(interval)
    }
  }, [taskId, uploadStatus, setUploadStatus])

  const handleUpload = useCallback(async () => {
    if (!selectedFile || uploadStatus === 'uploading') return

    setUploadStatus('uploading')
    setProgress({ step: '正在上传...', pct: 0 })

    try {
      const result = await uploadPDF(selectedFile)
      if (result.task_id) {
        setTaskId(result.task_id)
        setProgress({ step: '上传完成，等待处理...', pct: 5 })
      } else {
        setUploadStatus('success', result.filename)
        toast.success('上传成功！')
      }
      setSelectedFile(null)
    } catch (err) {
      setUploadStatus('error')
      toast.error('上传失败', {
        description: (err as Error).message || '请检查后端是否已启动',
      })
    }
  }, [selectedFile, uploadStatus, setUploadStatus])

  const handleFileSelect = useCallback(
    (file: File) => {
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        toast.warning('仅支持 PDF 文件')
        return
      }
      setSelectedFile(file)
      setUploadStatus('idle')
      setTaskId(null)
      setProgress({ step: '', pct: 0 })
    },
    [setUploadStatus]
  )

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault()
      setIsDragOver(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFileSelect(file)
    },
    [handleFileSelect]
  )

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFileSelect(file)
      e.target.value = ''
    },
    [handleFileSelect]
  )

  return (
    <div className="space-y-2">
      {/* 拖拽区域 */}
      <div
        className={`rounded-lg border-2 border-dashed p-4 text-center cursor-pointer transition-colors hover:border-primary/40 hover:bg-primary/[0.02] ${isDragOver ? 'border-primary bg-primary/[0.04]' : 'border-border'}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => document.getElementById('pdf-upload-input')?.click()}
      >
        {uploadStatus === 'uploading' ? (
          <div className="space-y-2">
            <Loader2 className="h-5 w-5 animate-spin text-primary mx-auto" />
            <p className="text-xs font-medium">{progress.step || '处理中...'}</p>
            {progress.pct > 0 && (
              <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all duration-500"
                  style={{ width: `${progress.pct}%` }}
                />
              </div>
            )}
          </div>
        ) : (
          <>
            <Upload className="h-5 w-5 text-muted-foreground mx-auto mb-1.5" />
            <p className="text-xs font-medium">上传 PDF 财报</p>
            <p className="text-[10px] text-muted-foreground mt-0.5">
              拖拽文件或点击选择
            </p>
          </>
        )}
        <input
          id="pdf-upload-input"
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={handleInputChange}
        />
      </div>

      {/* 已选文件 */}
      {selectedFile && (
        <div className="flex items-start gap-2 rounded-md border bg-card px-2.5 py-2">
          <FileText className="h-4 w-4 text-primary shrink-0" />
          <span className="min-w-0 flex-1 break-all text-xs leading-relaxed" title={selectedFile.name}>{selectedFile.name}</span>
          <Button
            size="sm"
            className="h-7 shrink-0 text-xs"
            onClick={(event) => {
              event.stopPropagation()
              void handleUpload()
            }}
            disabled={uploadStatus === 'uploading'}
          >
            入库
          </Button>
        </div>
      )}

      {/* 上次成功 */}
      {uploadStatus === 'success' && !selectedFile && lastUploadFilename && (
        <div className="flex items-center gap-2 rounded-md bg-success-light px-2.5 py-2 text-xs text-success">
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate">{lastUploadFilename} 已入库</span>
        </div>
      )}

      {/* 失败提示 */}
      {uploadStatus === 'error' && !selectedFile && (
        <div className="flex items-center gap-2 rounded-md bg-danger-light px-2.5 py-2 text-xs text-danger">
          <XCircle className="h-3.5 w-3.5 shrink-0" />
          <span>入库失败，请检查后端日志后重试</span>
        </div>
      )}
    </div>
  )
}
