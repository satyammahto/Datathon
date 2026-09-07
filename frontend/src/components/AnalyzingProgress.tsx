import { useEffect, useState } from 'react';
import { CheckCircle2, Sparkles, AlertCircle } from 'lucide-react';
import clsx from 'clsx';
import { fetchWithAuth, API_BASE_URL } from '../utils/apiClient';
import { useStore } from '../store/useStore';

interface AnalyzingProgressProps {
  datasetId?: string | null;
  onComplete: () => void;
}

export function AnalyzingProgress({ datasetId, onComplete }: AnalyzingProgressProps) {
  const [percent, setPercent] = useState(15);
  const [activeStep, setActiveStep] = useState(0);
  const [currentLabel, setCurrentLabel] = useState('Initializing pipeline...');
  const [error, setError] = useState<string | null>(null);
  const setDashboardContract = useStore(state => state.setDashboardContract);
  const setPipelineProgress = useStore(state => state.setPipelineProgress);

  const steps = [
    { key: 'discovery', title: 'Dataset Discovery & Profiling' },
    { key: 'quality_audit', title: 'Data Quality & Leakage Audit' },
    { key: 'model_championship', title: 'Model Championship Benchmark' },
    { key: 'insight_investigation', title: 'Multi-Perspective Investigation' },
    { key: 'fact_verification', title: 'Cross-Examination & Verification' },
    { key: 'dashboard_assembly', title: 'Dashboard Contract Assembly' }
  ];

  useEffect(() => {
    let isMounted = true;

    if (!datasetId) {
      // Fallback timer if no dataset ID provided
      const timer = setInterval(() => {
        setPercent((prev) => {
          if (prev >= 100) {
            clearInterval(timer);
            setTimeout(() => {
              if (isMounted) onComplete();
            }, 600);
            return 100;
          }
          const next = prev + 18;
          const stepIdx = Math.min(steps.length - 1, Math.floor((next / 100) * steps.length));
          setActiveStep(stepIdx);
          setCurrentLabel(steps[stepIdx].title);
          return Math.min(100, next);
        });
      }, 500);

      return () => {
        isMounted = false;
        clearInterval(timer);
      };
    }

    // Real-time backend status polling
    let pollCount = 0;
    const interval = setInterval(async () => {
      pollCount++;
      try {
        const res = await fetchWithAuth(`${API_BASE_URL}/api/pipeline/status/${datasetId}`);
        if (!res.ok) {
          // If status not ready yet, keep progressing gently
          setPercent(prev => Math.min(90, prev + 5));
          return;
        }

        const data = await res.json();
        if (!isMounted) return;

        setPipelineProgress(data);

        if (data.overall_progress) {
          setPercent(data.overall_progress);
        }

        if (data.current_stage) {
          const idx = steps.findIndex(s => s.key === data.current_stage);
          if (idx !== -1) {
            setActiveStep(idx);
            setCurrentLabel(steps[idx].title);
          }
        }

        // Check if finished
        if (!data.is_running && data.overall_progress >= 100) {
          clearInterval(interval);
          // Fetch final dashboard contract
          try {
            const dashRes = await fetchWithAuth(`${API_BASE_URL}/api/pipeline/dashboard/${datasetId}`);
            if (dashRes.ok) {
              const dashData = await dashRes.json();
              setDashboardContract(dashData);
            }
          } catch (err) {
            console.warn('Dashboard contract fetch error:', err);
          }

          setTimeout(() => {
            if (isMounted) onComplete();
          }, 600);
        } else if (pollCount > 60) {
          // Timeout after 60s
          clearInterval(interval);
          setError('Pipeline took longer than expected. Proceeding to dashboard.');
          setTimeout(() => {
            if (isMounted) onComplete();
          }, 1500);
        }
      } catch (err) {
        console.warn('Status poll error', err);
      }
    }, 800);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [datasetId, onComplete, setDashboardContract, setPipelineProgress]);

  // Circular gauge math (radius = 54, strokeWidth = 8, circumference = 2 * PI * 54 = ~339.29)
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  return (
    <div className="flex flex-col items-center justify-center max-w-xl mx-auto py-8 px-4 text-center space-y-7 animate-in fade-in duration-300">
      <div className="space-y-2">
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
          Analyzing your data with AIDA
        </h2>
        <p className="text-xs sm:text-sm text-slate-400 max-w-md mx-auto leading-relaxed">
          {currentLabel}
        </p>
      </div>

      {error && (
        <div className="p-2.5 bg-amber-500/10 border border-amber-500/20 rounded-xl text-amber-400 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Circular Progress Gauge */}
      <div className="relative w-36 h-36 flex items-center justify-center">
        <svg className="w-full h-full -rotate-90 transform" viewBox="0 0 128 128">
          <circle
            cx="64"
            cy="64"
            r={radius}
            className="text-[#131b2e]"
            strokeWidth="8"
            stroke="currentColor"
            fill="transparent"
          />
          <circle
            cx="64"
            cy="64"
            r={radius}
            className="transition-all duration-500 ease-out"
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            stroke="url(#progressGradient)"
            fill="transparent"
          />
          <defs>
            <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#38bdf8" />
              <stop offset="100%" stopColor="#6366f1" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-black text-white tracking-tight">{percent}%</span>
        </div>
      </div>

      {/* 6 Step Checklist */}
      <div className="w-full max-w-md bg-[#0f172a]/60 border border-[#1e293b] rounded-2xl p-5 space-y-3 text-left shadow-lg">
        {steps.map((step, idx) => {
          const isDone = idx < activeStep || percent === 100;
          const isCurrent = idx === activeStep && percent < 100;
          return (
            <div key={idx} className="flex items-center gap-3 text-xs">
              {isDone ? (
                <div className="w-4 h-4 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center flex-shrink-0">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                </div>
              ) : isCurrent ? (
                <div className="w-4 h-4 rounded-full border-2 border-sky-400 border-t-transparent animate-spin flex-shrink-0" />
              ) : (
                <div className="w-4 h-4 rounded-full border border-slate-600 flex-shrink-0" />
              )}
              <span
                className={clsx(
                  "transition-colors",
                  isDone && "text-slate-300",
                  isCurrent && "text-sky-300 font-semibold",
                  !isDone && !isCurrent && "text-slate-500"
                )}
              >
                {step.title}
              </span>
            </div>
          );
        })}
      </div>

      {/* Bottom Sparkle Quote Box */}
      <div className="bg-[#111c35]/60 border border-[#1e293b] rounded-xl px-5 py-2.5 flex items-center gap-2 text-xs text-indigo-300 shadow-sm">
        <Sparkles className="w-3.5 h-3.5 text-indigo-400 flex-shrink-0" />
        <span className="italic">"Continuous 4-brain intelligence running across all stages."</span>
      </div>
    </div>
  );
}
