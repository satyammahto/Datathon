import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  UploadCloud,
  Sparkles,
  CheckCircle2,
  Trash2,
  ArrowRight,
  BarChart2,
  TrendingUp,
  MessageSquare,
  Wand2,
  FileSpreadsheet,
  AlertCircle,
  Database
} from 'lucide-react';
import { AnalyzingProgress } from '../components/AnalyzingProgress';
import { useStore } from '../store/useStore';
import { fetchWithAuth, API_BASE_URL } from '../utils/apiClient';

interface SelectedFile {
  id: string;
  name: string;
  size: string;
  type: string;
  fileObj?: File;
}

export function NewAnalysis() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadedDatasetId, setUploadedDatasetId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const setActiveDatasetId = useStore(state => state.setActiveDatasetId);
  const setActiveScenario = useStore(state => state.setActiveScenario);

  const [files, setFiles] = useState<SelectedFile[]>([]);

  const handleRemoveFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const handleClearAll = () => {
    setFiles([]);
    setErrorMessage(null);
  };

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const newFiles: SelectedFile[] = Array.from(e.dataTransfer.files).map((f, i) => ({
        id: `drop-${Date.now()}-${i}`,
        name: f.name,
        size: `${(f.size / (1024 * 1024)).toFixed(1)} MB`,
        type: f.name.split('.').pop()?.toUpperCase() || 'DATA',
        fileObj: f
      }));
      setFiles((prev) => [...prev, ...newFiles]);
      setErrorMessage(null);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles: SelectedFile[] = Array.from(e.target.files).map((f, i) => ({
        id: `upload-${Date.now()}-${i}`,
        name: f.name,
        size: `${(f.size / (1024 * 1024)).toFixed(1)} MB`,
        type: f.name.split('.').pop()?.toUpperCase() || 'DATA',
        fileObj: f
      }));
      setFiles((prev) => [...prev, ...newFiles]);
      setErrorMessage(null);
    }
  };

  const handleSelectSample = (sample: 'churn' | 'sales') => {
    if (sample === 'churn') {
      setActiveScenario('churn');
      setActiveDatasetId('ds-churn-901');
      setUploadedDatasetId('ds-churn-901');
      setIsAnalyzing(true);
    } else {
      setActiveScenario('sales');
      setActiveDatasetId('ds-sales-502');
      setUploadedDatasetId('ds-sales-502');
      setIsAnalyzing(true);
    }
  };

  const handleRunAnalysis = async () => {
    if (files.length === 0) return;
    setErrorMessage(null);

    const primaryFile = files[0];
    if (!primaryFile.fileObj) {
      // Fallback demo run
      setActiveDatasetId('ds-churn-901');
      setUploadedDatasetId('ds-churn-901');
      setIsAnalyzing(true);
      return;
    }

    try {
      setIsAnalyzing(true);
      const formData = new FormData();
      formData.append('file', primaryFile.fileObj);

      const res = await fetchWithAuth(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Upload and analysis initiation failed');
      }

      const dataset = await res.json();
      setActiveDatasetId(dataset.id);
      setUploadedDatasetId(dataset.id);

      // Explicitly trigger the live pipeline run
      fetchWithAuth(`${API_BASE_URL}/api/pipeline/run/${dataset.id}`, { method: 'POST' })
        .catch(err => console.warn('Background trigger note:', err));

    } catch (err: any) {
      setIsAnalyzing(false);
      setErrorMessage(err.message || 'Failed to analyze file');
    }
  };

  const handleAnalysisComplete = () => {
    setIsAnalyzing(false);
    navigate('/');
  };

  if (isAnalyzing) {
    return (
      <AnalyzingProgress
        datasetId={uploadedDatasetId}
        onComplete={handleAnalysisComplete}
      />
    );
  }

  return (
    <div className="max-w-4xl mx-auto py-4 px-4 space-y-7 animate-in fade-in duration-200">
      {/* Top Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-[#171f38] text-indigo-300 border border-indigo-500/20 shadow-sm">
          <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
          <span>Autonomous 4-Brain Data Analyst</span>
        </div>

        <h1 className="text-3xl sm:text-4xl font-black tracking-tight text-white">
          Automated Insight Analyst
        </h1>
        <p className="text-xs sm:text-sm text-slate-400">
          Upload any CSV or Excel file. AIDA discovers, benchmarks, verifies, and visualizes the story.
        </p>
      </div>

      {errorMessage && (
        <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* 4 Feature Pills Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        <div className="bg-[#0f172a] border border-[#1e293b] rounded-xl py-2 px-3 flex items-center justify-center gap-2 text-xs text-slate-300 shadow-sm">
          <Wand2 className="w-3.5 h-3.5 text-sky-400" />
          <span className="font-medium">Automatic data cleaning</span>
        </div>
        <div className="bg-[#0f172a] border border-[#1e293b] rounded-xl py-2 px-3 flex items-center justify-center gap-2 text-xs text-slate-300 shadow-sm">
          <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
          <span className="font-medium">ML Championship</span>
        </div>
        <div className="bg-[#0f172a] border border-[#1e293b] rounded-xl py-2 px-3 flex items-center justify-center gap-2 text-xs text-slate-300 shadow-sm">
          <BarChart2 className="w-3.5 h-3.5 text-purple-400" />
          <span className="font-medium">Beautiful visualizations</span>
        </div>
        <div className="bg-[#0f172a] border border-[#1e293b] rounded-xl py-2 px-3 flex items-center justify-center gap-2 text-xs text-slate-300 shadow-sm">
          <MessageSquare className="w-3.5 h-3.5 text-amber-400" />
          <span className="font-medium">Multi-agent verification</span>
        </div>
      </div>

      {/* Big Drag and Drop Box */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleFileDrop}
        onClick={() => fileInputRef.current?.click()}
        className="bg-[#0e1629]/70 border-2 border-dashed border-[#22304f] hover:border-indigo-500/50 rounded-2xl p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-all hover:bg-[#121c33]/80 group shadow-lg"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls,.json"
          className="hidden"
          onChange={handleFileInputChange}
        />

        <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-indigo-500/20 to-sky-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 mb-3 shadow-inner group-hover:scale-105 transition-transform">
          <UploadCloud className="w-8 h-8" />
        </div>

        <h3 className="text-base font-bold text-white tracking-tight">
          Drop your dataset here
        </h3>
        <p className="text-xs text-indigo-400 mt-0.5">
          or click to browse from your computer
        </p>
        <span className="text-[11px] text-slate-500 mt-2">
          Supports CSV, XLSX, XLS, and JSON datasets
        </span>
      </div>

      {/* Sample Datasets Quick Access */}
      <div className="flex items-center justify-between p-3.5 bg-[#0f172a]/60 border border-[#1e293b] rounded-xl text-xs">
        <div className="flex items-center gap-2 text-slate-300">
          <Database className="w-4 h-4 text-indigo-400" />
          <span className="font-medium">Or explore standard benchmark scenarios:</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => handleSelectSample('churn')}
            className="px-2.5 py-1 rounded-lg bg-[#171f38] hover:bg-[#1f2b4d] text-indigo-300 border border-indigo-500/30 font-semibold text-[11px] transition-colors"
          >
            Customer Churn (Classification)
          </button>
          <button
            type="button"
            onClick={() => handleSelectSample('sales')}
            className="px-2.5 py-1 rounded-lg bg-[#171f38] hover:bg-[#1f2b4d] text-indigo-300 border border-indigo-500/30 font-semibold text-[11px] transition-colors"
          >
            Retail Sales (Time Series)
          </button>
        </div>
      </div>

      {/* Selected Files Queue */}
      {files.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400 px-1">
            <span className="font-semibold text-slate-300">{files.length} file selected for analysis</span>
            <button
              onClick={handleClearAll}
              type="button"
              className="text-slate-400 hover:text-white transition-colors"
            >
              Clear
            </button>
          </div>

          <div className="space-y-2">
            {files.map((file) => (
              <div
                key={file.id}
                className="bg-[#0f172a] border border-[#1e293b] rounded-xl px-4 py-3 flex items-center justify-between text-xs shadow-sm hover:border-slate-700 transition-colors"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
                    <FileSpreadsheet className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <p className="font-semibold text-white truncate">{file.name}</p>
                    <p className="text-[10px] text-slate-400 mt-0.5">
                      {file.type} &bull; {file.size}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="w-5 h-5 rounded-full bg-emerald-500/10 text-emerald-400 flex items-center justify-center">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  </div>
                  <button
                    onClick={() => handleRemoveFile(file.id)}
                    type="button"
                    className="text-slate-500 hover:text-red-400 p-1 transition-colors"
                    title="Remove file"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* Large Gradient Run Analysis Button */}
          <div className="pt-2">
            <button
              onClick={handleRunAnalysis}
              type="button"
              className="w-full bg-gradient-to-r from-[#6366f1] via-[#4f46e5] to-[#3b82f6] hover:from-[#5558e6] hover:to-[#2563eb] text-white font-bold py-3.5 rounded-xl text-sm shadow-xl shadow-indigo-500/25 flex items-center justify-center gap-2 transition-all group"
            >
              <span>Launch Live Analysis Pipeline</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
