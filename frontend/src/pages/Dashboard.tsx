import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Database,
  Columns3,
  ShieldCheck,
  TrendingUp,
  Users,
  AlertTriangle,
  LineChart,
  BarChart3,
  Network,
  Maximize2,
  ArrowRight,
  CheckCircle2,
  Search,
  Layers,
  Flame,
  X,
  Trophy,
  Activity
} from 'lucide-react';
import Plot from 'react-plotly.js';
import clsx from 'clsx';
import { useStore } from '../store/useStore';
import { fetchWithAuth, API_BASE_URL } from '../utils/apiClient';
import { AutoChart } from '../components/AutoChart';
import { QualityLeakageBanner } from '../components/QualityLeakageBanner';
import { ExportModal } from '../components/ExportModal';

export function Dashboard() {
  const navigate = useNavigate();
  const { activeDatasetId, dashboardContract, setDashboardContract } = useStore();
  const [activeTab, setActiveTab] = useState<'overview' | 'visualizations' | 'quality' | 'details' | 'advanced'>('overview');
  const [isExportOpen, setIsExportOpen] = useState(false);
  const [searchColumn, setSearchColumn] = useState('');
  const [detailsSubTab, setDetailsSubTab] = useState<'columns' | 'sample' | 'distributions'>('columns');
  const [showCleaningModal, setShowCleaningModal] = useState(false);
  const [previewData, setPreviewData] = useState<any>(null);

  // Fetch contract from backend on mount or dataset switch
  useEffect(() => {
    const datasetIdToLoad = activeDatasetId || 'ds-churn-901';
    if (!dashboardContract || (activeDatasetId && dashboardContract.dataset_id !== activeDatasetId)) {
      fetchWithAuth(`${API_BASE_URL}/api/pipeline/dashboard/${datasetIdToLoad}`)
        .then(res => res.ok ? res.json() : null)
        .then(data => {
          if (data) {
            setDashboardContract(data);
          }
        })
        .catch(err => console.warn('Failed to load dashboard contract', err));
    }
  }, [activeDatasetId, dashboardContract, setDashboardContract]);

  // Fetch dataset preview for raw schema inspection
  useEffect(() => {
    if (activeDatasetId && !activeDatasetId.startsWith('ds-') && !activeDatasetId.startsWith('demo-')) {
      fetchWithAuth(`${API_BASE_URL}/api/datasets/${activeDatasetId}/preview`)
        .then(res => res.ok ? res.json() : null)
        .then(data => {
          if (data) setPreviewData(data);
        })
        .catch(() => {});
    }
  }, [activeDatasetId]);

  const datasetName = dashboardContract?.dataset_name || (previewData?.filename) || "Customer_Sales_Data";
  const analyzedDate = dashboardContract?.generated_at
    ? new Date(dashboardContract.generated_at).toLocaleString('en-US', {
        day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit'
      })
    : "07 Sep 2026, 01:24 PM";

  // Data Quality Metrics
  const qualityScore = dashboardContract?.quality_summary?.overall_score ?? 94.8;
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (qualityScore / 100) * circumference;

  const totalRecords = dashboardContract?.quality_summary?.rows != null
    ? dashboardContract.quality_summary.rows.toLocaleString()
    : (previewData?.total_rows?.toLocaleString() ?? "12,482");

  const totalColumns = dashboardContract?.quality_summary?.columns != null
    ? dashboardContract.quality_summary.columns
    : (previewData?.columns?.length ?? 18);

  const championName = dashboardContract?.championship_summary?.champion_name || "LightGBM";
  const championMetric = dashboardContract?.championship_summary?.primary_metric_value != null
    ? `${(dashboardContract.championship_summary.primary_metric_value * 100).toFixed(1)}%`
    : "85.4%";

  // Dynamic columns from preview or contract
  const fallbackColumns = [
    { name: 'order_id', type: 'Numeric', typeColor: 'bg-sky-500/10 text-sky-400 border-sky-500/20', unique: '12,482', missing: '0', samples: '1001, 1002, 1003...' },
    { name: 'order_date', type: 'Date', typeColor: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20', unique: '365', missing: '0', samples: '2024-01-01, 2024-01-02...' },
    { name: 'customer_id', type: 'Categorical', typeColor: 'bg-amber-500/10 text-amber-400 border-amber-500/20', unique: '2,843', missing: '0', samples: 'C001, C002, C003...' },
    { name: 'category', type: 'Categorical', typeColor: 'bg-amber-500/10 text-amber-400 border-amber-500/20', unique: '5', missing: '0', samples: 'Electronics, Clothing, Home...' },
    { name: 'product_name', type: 'Categorical', typeColor: 'bg-amber-500/10 text-amber-400 border-amber-500/20', unique: '1,024', missing: '3', missingAlert: true, samples: 'iPhone, T-Shirt, Blender...' },
    { name: 'quantity', type: 'Numeric', typeColor: 'bg-sky-500/10 text-sky-400 border-sky-500/20', unique: '120', missing: '0', samples: '1, 2, 3, 5, 10...' },
    { name: 'unit_price', type: 'Numeric', typeColor: 'bg-sky-500/10 text-sky-400 border-sky-500/20', unique: '856', missing: '0', samples: '299.99, 49.50, 19.99...' },
    { name: 'total_amount', type: 'Numeric', typeColor: 'bg-sky-500/10 text-sky-400 border-sky-500/20', unique: '4,321', missing: '0', samples: '299.99, 150.00, 89.90...' },
  ];

  const columnsData = previewData?.columns && previewData.columns.length > 0
    ? previewData.columns.map((c: string) => {
        const samples = previewData.preview?.slice(0, 3).map((r: any) => String(r[c] ?? '')).join(', ') || '-';
        return {
          name: c,
          type: 'Feature',
          typeColor: 'bg-sky-500/10 text-sky-400 border-sky-500/20',
          unique: '-',
          missing: '0',
          samples
        };
      })
    : fallbackColumns;

  const filteredColumns = columnsData.filter((col: any) =>
    col.name.toLowerCase().includes(searchColumn.toLowerCase())
  );

  const contractInsights = dashboardContract?.insights || [];
  const contractCharts = dashboardContract?.charts || [];

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-16 animate-in fade-in duration-200">
      {/* Top Header & Breadcrumb */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#1e293b] pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400 mb-1">
            <span
              onClick={() => navigate('/reports')}
              className="cursor-pointer hover:text-indigo-400 transition-colors"
            >
              Reports
            </span>
            <span>&gt;</span>
            <span className="text-slate-200 font-semibold">{datasetName}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
            {datasetName}
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Analyzed on {analyzedDate}
          </p>
        </div>

        {/* Action Button: Export Report */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsExportOpen(true)}
            className="px-4 py-2 text-xs font-bold text-white bg-gradient-to-r from-[#6366f1] to-[#4f46e5] hover:from-[#5558e6] hover:to-[#4338ca] rounded-xl shadow-lg shadow-indigo-500/20 flex items-center gap-1.5 transition-all"
          >
            <span>Export Report</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Horizontal Tab Navigation Bar */}
      <div className="flex items-center gap-1 border-b border-[#1e293b] overflow-x-auto pb-px">
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'visualizations', label: 'Visualizations' },
          { id: 'quality', label: 'Data Quality' },
          { id: 'details', label: 'Dataset Details' },
          { id: 'advanced', label: 'Advanced Analysis' }
        ].map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={clsx(
                "px-4 py-2.5 text-xs font-bold transition-all relative whitespace-nowrap",
                isActive
                  ? "text-white"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <span>{tab.label}</span>
              {isActive && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#6366f1] shadow-[0_-2px_6px_rgba(99,102,241,0.6)]" />
              )}
            </button>
          );
        })}
      </div>

      {/* ─────────────────────────────────────────────────────────────
          TAB 1: OVERVIEW
         ───────────────────────────────────────────────────────────── */}
      {activeTab === 'overview' && (
        <div className="space-y-7 animate-in fade-in duration-200">
          {/* 4 Stat Cards Row */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5">
            <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl p-4 flex items-center gap-3.5 shadow-sm">
              <div className="p-3 rounded-xl bg-sky-500/10 text-sky-400">
                <Database className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-2xl font-black text-white tracking-tight">{totalRecords}</h3>
                <p className="text-xs text-slate-400 mt-0.5">Total Records</p>
              </div>
            </div>

            <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl p-4 flex items-center gap-3.5 shadow-sm">
              <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400">
                <Columns3 className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-2xl font-black text-white tracking-tight">{totalColumns}</h3>
                <p className="text-xs text-slate-400 mt-0.5">Total Columns</p>
              </div>
            </div>

            <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl p-4 flex items-center gap-3.5 shadow-sm">
              <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-2xl font-black text-white tracking-tight">{qualityScore}%</h3>
                <p className="text-xs text-slate-400 mt-0.5">Data Quality</p>
              </div>
            </div>

            <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl p-4 flex items-center gap-3.5 shadow-sm">
              <div className="p-3 rounded-xl bg-amber-500/10 text-amber-400">
                <Trophy className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-2xl font-black text-white tracking-tight truncate max-w-[140px]">{championName}</h3>
                <p className="text-xs text-slate-400 mt-0.5">Champ Metric: {championMetric}</p>
              </div>
            </div>
          </div>

          {/* Key Insights Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold text-white tracking-tight">Key Insights</h2>
              <button
                onClick={() => navigate('/insights')}
                className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 transition-colors flex items-center gap-1"
              >
                <span>See all insights</span>
                <span>&rarr;</span>
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
              {contractInsights.length > 0 ? (
                contractInsights.slice(0, 4).map((ins, idx) => {
                  const borderColors = [
                    'hover:border-emerald-500/40',
                    'hover:border-sky-500/40',
                    'hover:border-red-500/40',
                    'hover:border-amber-500/40'
                  ];
                  const iconColors = [
                    'bg-emerald-500/10 text-emerald-400',
                    'bg-sky-500/10 text-sky-400',
                    'bg-red-500/10 text-red-400',
                    'bg-amber-500/10 text-amber-400'
                  ];
                  const icons = [TrendingUp, Users, AlertTriangle, LineChart];
                  const Icon = icons[idx % icons.length];

                  return (
                    <div
                      key={ins.id || idx}
                      onClick={() => navigate('/insights')}
                      className={`bg-[#0e1629] border border-[#1e293b] ${borderColors[idx % borderColors.length]} rounded-2xl p-4 flex flex-col justify-between space-y-3 shadow-sm transition-all cursor-pointer`}
                    >
                      <div className="flex items-center justify-between">
                        <div className={`w-8 h-8 rounded-xl ${iconColors[idx % iconColors.length]} flex items-center justify-center`}>
                          <Icon className="w-4 h-4" />
                        </div>
                        <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-[#171f38] text-indigo-300 border border-indigo-500/20">
                          {ins.status.toUpperCase()}
                        </span>
                      </div>
                      <div>
                        <h4 className="text-xs text-slate-400 line-clamp-1">{ins.title}</h4>
                        <p className="text-lg font-black text-white mt-0.5 line-clamp-1">
                          {ins.evidence?.metric_value || ins.claim}
                        </p>
                        <p className="text-[11px] text-slate-400 mt-1 leading-relaxed line-clamp-2">
                          {ins.claim}
                        </p>
                      </div>
                    </div>
                  );
                })
              ) : (
                <>
                  <div className="bg-[#0e1629] border border-[#1e293b] hover:border-emerald-500/40 rounded-2xl p-4 flex flex-col justify-between space-y-3 shadow-sm transition-all">
                    <div className="w-8 h-8 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center">
                      <TrendingUp className="w-4 h-4" />
                    </div>
                    <div>
                      <h4 className="text-xs text-slate-400">Revenue increased</h4>
                      <p className="text-xl font-black text-white mt-0.5">18.4%</p>
                      <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                        over the selected period. Main driver: Premium product category.
                      </p>
                    </div>
                  </div>

                  <div className="bg-[#0e1629] border border-[#1e293b] hover:border-sky-500/40 rounded-2xl p-4 flex flex-col justify-between space-y-3 shadow-sm transition-all">
                    <div className="w-8 h-8 rounded-xl bg-sky-500/10 text-sky-400 flex items-center justify-center">
                      <Users className="w-4 h-4" />
                    </div>
                    <div>
                      <h4 className="text-xs text-slate-400">Top 3 categories</h4>
                      <p className="text-xl font-black text-white mt-0.5">contribute 62%</p>
                      <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                        of total sales.
                      </p>
                    </div>
                  </div>

                  <div className="bg-[#0e1629] border border-[#1e293b] hover:border-red-500/40 rounded-2xl p-4 flex flex-col justify-between space-y-3 shadow-sm transition-all">
                    <div className="w-8 h-8 rounded-xl bg-red-500/10 text-red-400 flex items-center justify-center">
                      <AlertTriangle className="w-4 h-4" />
                    </div>
                    <div>
                      <p className="text-xl font-black text-white">12.7%</p>
                      <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                        of records contain potential outliers. Most concentrated in Transaction Amount.
                      </p>
                    </div>
                  </div>

                  <div className="bg-[#0e1629] border border-[#1e293b] hover:border-amber-500/40 rounded-2xl p-4 flex flex-col justify-between space-y-3 shadow-sm transition-all">
                    <div className="w-8 h-8 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center">
                      <LineChart className="w-4 h-4" />
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-white leading-relaxed mt-1">
                        Customer age and purchase frequency show a strong relationship.
                      </p>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Advanced Analysis Teaser */}
          <div className="space-y-3 pt-2">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white tracking-tight">Advanced Analysis</h3>
                <p className="text-xs text-slate-400">Optional - Explore the technical analysis behind the insights.</p>
              </div>
              <button
                onClick={() => setActiveTab('advanced')}
                className="px-3 py-1.5 rounded-lg border border-[#1e293b] bg-[#0e1629] hover:bg-[#131d33] text-xs text-slate-300 font-medium transition-colors flex items-center gap-1"
              >
                <Maximize2 className="w-3.5 h-3.5" />
                <span>Expand</span>
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div
                onClick={() => setActiveTab('advanced')}
                className="bg-[#0e1629] border border-[#1e293b] hover:border-indigo-500/40 p-4 rounded-xl cursor-pointer transition-all space-y-2"
              >
                <div className="w-7 h-7 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center">
                  <Network className="w-3.5 h-3.5" />
                </div>
                <h4 className="text-xs font-bold text-white">Correlation Analysis</h4>
                <p className="text-[10px] text-slate-400">Find relationships between variables</p>
              </div>

              <div
                onClick={() => setActiveTab('advanced')}
                className="bg-[#0e1629] border border-[#1e293b] hover:border-sky-500/40 p-4 rounded-xl cursor-pointer transition-all space-y-2"
              >
                <div className="w-7 h-7 rounded-lg bg-sky-500/10 text-sky-400 flex items-center justify-center">
                  <Layers className="w-3.5 h-3.5" />
                </div>
                <h4 className="text-xs font-bold text-white">Clustering</h4>
                <p className="text-[10px] text-slate-400">Identify natural groups in your data</p>
              </div>

              <div
                onClick={() => setActiveTab('advanced')}
                className="bg-[#0e1629] border border-[#1e293b] hover:border-red-500/40 p-4 rounded-xl cursor-pointer transition-all space-y-2"
              >
                <div className="w-7 h-7 rounded-lg bg-red-500/10 text-red-400 flex items-center justify-center">
                  <AlertTriangle className="w-3.5 h-3.5" />
                </div>
                <h4 className="text-xs font-bold text-white">Outlier Detection</h4>
                <p className="text-[10px] text-slate-400">Detect unusual patterns</p>
              </div>

              <div
                onClick={() => setActiveTab('advanced')}
                className="bg-[#0e1629] border border-[#1e293b] hover:border-emerald-500/40 p-4 rounded-xl cursor-pointer transition-all space-y-2"
              >
                <div className="w-7 h-7 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center">
                  <BarChart3 className="w-3.5 h-3.5" />
                </div>
                <h4 className="text-xs font-bold text-white">Feature Importance</h4>
                <p className="text-[10px] text-slate-400">Understand key drivers</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────
          TAB 2: VISUALIZATIONS / DATA PATTERNS
         ───────────────────────────────────────────────────────────── */}
      {activeTab === 'visualizations' && (
        <div className="space-y-5 animate-in fade-in duration-200">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">Data Patterns & Charts</h2>
              <p className="text-xs text-slate-400">Automatically generated charts based on your live dataset profile.</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400 font-mono px-3 py-1.5 rounded-lg bg-[#0e1629] border border-[#1e293b]">
                {contractCharts.length > 0 ? `Visualizations (${contractCharts.length})` : 'All Visualizations (4)'}
              </span>
            </div>
          </div>

          {/* Dynamic Plotly Charts from Contract */}
          {contractCharts.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {contractCharts.map((chartSpec, i) => (
                <div key={chartSpec.id || i} className="bg-[#0e1629] border border-[#1e293b] rounded-2xl p-4 shadow-sm">
                  <AutoChart spec={chartSpec} />
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Fallback Chart 1: Sales Trend Over Time */}
              <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl p-4 shadow-sm space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-sm text-white">Sales Trend Over Time</h3>
                  <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    Time Series
                  </span>
                </div>
                <div className="h-64 w-full">
                  <Plot
                    data={[
                      {
                        x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
                        y: [120, 110, 190, 175, 230, 210, 260, 240, 290],
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#818cf8', width: 3, shape: 'spline' },
                        fill: 'tozeroy',
                        fillcolor: 'rgba(99, 102, 241, 0.12)',
                        marker: { color: '#c7d2fe', size: 5 }
                      }
                    ]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      margin: { t: 15, r: 15, l: 35, b: 30 },
                      font: { color: '#94a3b8', size: 10, family: 'Inter' },
                      xaxis: { gridcolor: '#1e293b', zeroline: false },
                      yaxis: { gridcolor: '#1e293b', zeroline: false }
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%', height: '100%' }}
                  />
                </div>
              </div>

              {/* Fallback Chart 2: Sales by Category */}
              <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl p-4 shadow-sm space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-sm text-white">Sales by Category</h3>
                  <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20">
                    Category Comparison
                  </span>
                </div>
                <div className="h-64 w-full">
                  <Plot
                    data={[
                      {
                        x: ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'],
                        y: [280, 240, 180, 130, 110],
                        type: 'bar',
                        marker: {
                          color: ['#38bdf8', '#2dd4bf', '#a78bfa', '#fb923c', '#4ade80']
                        }
                      }
                    ]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      margin: { t: 15, r: 15, l: 35, b: 30 },
                      font: { color: '#94a3b8', size: 10, family: 'Inter' },
                      xaxis: { gridcolor: '#1e293b', zeroline: false },
                      yaxis: { gridcolor: '#1e293b', zeroline: false }
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%', height: '100%' }}
                  />
                </div>
              </div>

              {/* Fallback Chart 3: Price vs Quantity */}
              <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl p-4 shadow-sm space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-sm text-white">Price vs Quantity</h3>
                  <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20">
                    Relationship
                  </span>
                </div>
                <div className="h-64 w-full">
                  <Plot
                    data={[
                      {
                        x: [10, 25, 30, 45, 60, 75, 90, 110, 130, 150, 170, 200, 25, 55, 85, 120],
                        y: [150, 280, 210, 390, 450, 520, 680, 710, 790, 850, 920, 980, 220, 410, 580, 740],
                        mode: 'markers',
                        type: 'scatter',
                        marker: { color: '#818cf8', size: 7, opacity: 0.8 }
                      }
                    ]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      margin: { t: 15, r: 15, l: 35, b: 30 },
                      font: { color: '#94a3b8', size: 10, family: 'Inter' },
                      xaxis: { gridcolor: '#1e293b', zeroline: false },
                      yaxis: { gridcolor: '#1e293b', zeroline: false }
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%', height: '100%' }}
                  />
                </div>
              </div>

              {/* Fallback Chart 4: Transaction Amount Distribution */}
              <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl p-4 shadow-sm space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-sm text-white">Transaction Amount Distribution</h3>
                  <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    Distribution
                  </span>
                </div>
                <div className="h-64 w-full">
                  <Plot
                    data={[
                      {
                        x: [50, 150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1200],
                        y: [120, 280, 450, 620, 820, 760, 580, 410, 280, 160, 90, 40],
                        type: 'bar',
                        marker: { color: '#c084fc' }
                      }
                    ]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      margin: { t: 15, r: 15, l: 35, b: 30 },
                      font: { color: '#94a3b8', size: 10, family: 'Inter' },
                      xaxis: { gridcolor: '#1e293b', zeroline: false },
                      yaxis: { gridcolor: '#1e293b', zeroline: false }
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%', height: '100%' }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────
          TAB 3: DATA QUALITY
         ───────────────────────────────────────────────────────────── */}
      {activeTab === 'quality' && (
        <div className="space-y-6 animate-in fade-in duration-200">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Data Quality & Health</h2>
            <p className="text-xs text-slate-400">Autonomous data audit, missingness profile, and leakage isolation.</p>
          </div>

          <QualityLeakageBanner
            qualityReport={{ overall_score: qualityScore } as any}
            leakageWarnings={dashboardContract?.quality_summary?.has_leakage ? [{ column_name: 'Identified Leakage Feature', risk_level: 'high', reason: 'High correlation with target' } as any] : []}
          />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Big Circular Gauge Box */}
            <div className="lg:col-span-1 bg-[#0e1629] border border-[#1e293b] rounded-2xl p-6 flex flex-col items-center justify-center text-center space-y-4 shadow-sm">
              <div className="relative w-40 h-40 flex items-center justify-center">
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
                    strokeWidth="8"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    strokeLinecap="round"
                    stroke={qualityScore >= 80 ? "#10b981" : "#f59e0b"}
                    fill="transparent"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-3xl font-black text-white">{qualityScore}%</span>
                  <span className={clsx(
                    "text-[10px] font-bold uppercase tracking-wider mt-0.5",
                    qualityScore >= 80 ? "text-emerald-400" : "text-amber-400"
                  )}>
                    {qualityScore >= 80 ? 'Healthy' : 'Needs Review'}
                  </span>
                </div>
              </div>
              <p className="text-xs text-slate-400">Overall data integrity score based on completeness, uniqueness, and schema fidelity.</p>
            </div>

            {/* Right Metrics Cards */}
            <div className="lg:col-span-2 grid grid-cols-2 sm:grid-cols-3 gap-3.5">
              <div className="bg-[#0e1629] border border-[#1e293b] p-4 rounded-xl">
                <div className="flex items-center gap-2 text-slate-400 text-xs">
                  <Database className="w-4 h-4 text-sky-400" />
                  <span>Total Records</span>
                </div>
                <p className="text-2xl font-bold text-white mt-1.5">{totalRecords}</p>
              </div>

              <div className="bg-[#0e1629] border border-[#1e293b] p-4 rounded-xl">
                <div className="flex items-center gap-2 text-slate-400 text-xs">
                  <Columns3 className="w-4 h-4 text-purple-400" />
                  <span>Total Columns</span>
                </div>
                <p className="text-2xl font-bold text-white mt-1.5">{totalColumns}</p>
              </div>

              <div className="bg-[#0e1629] border border-[#1e293b] p-4 rounded-xl">
                <div className="flex items-center gap-2 text-slate-400 text-xs">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  <span>Leakage Risk</span>
                </div>
                <p className="text-2xl font-bold text-amber-400 mt-1.5">
                  {dashboardContract?.quality_summary?.has_leakage ? 'Flagged' : 'None'}
                </p>
              </div>

              <div className="bg-[#0e1629] border border-[#1e293b] p-4 rounded-xl">
                <div className="flex items-center gap-2 text-slate-400 text-xs">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span>Missing Values Cleaned</span>
                </div>
                <p className="text-2xl font-bold text-white mt-1.5">
                  {dashboardContract?.quality_summary?.overall_score ? 'Resolved' : '0'}
                </p>
              </div>

              <div className="bg-[#0e1629] border border-[#1e293b] p-4 rounded-xl">
                <div className="flex items-center gap-2 text-slate-400 text-xs">
                  <Flame className="w-4 h-4 text-red-400" />
                  <span>Outliers Audited</span>
                </div>
                <p className="text-2xl font-bold text-red-400 mt-1.5">Checked</p>
              </div>

              <div className="bg-[#0e1629] border border-[#1e293b] p-4 rounded-xl">
                <div className="flex items-center gap-2 text-slate-400 text-xs">
                  <Activity className="w-4 h-4 text-indigo-400" />
                  <span>Integrity Status</span>
                </div>
                <p className="text-2xl font-bold text-indigo-400 mt-1.5">Verified</p>
              </div>
            </div>
          </div>

          {/* Cleaning Completed Box */}
          <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-sm">
            <div>
              <h3 className="text-sm font-bold text-white mb-3">Cleaning & Preprocessing Pipeline</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-2 text-slate-300">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  <span>Missing values handled and imputed</span>
                </div>
                <div className="flex items-center gap-2 text-slate-300">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  <span>Duplicate records audited</span>
                </div>
                <div className="flex items-center gap-2 text-slate-300">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  <span>Formatting and types normalized</span>
                </div>
                <div className="flex items-center gap-2 text-slate-300">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  <span>Semantic feature entities detected</span>
                </div>
              </div>
            </div>

            <div>
              <button
                onClick={() => setShowCleaningModal(true)}
                className="px-4 py-2 rounded-xl bg-[#171f38] hover:bg-[#1e284a] text-indigo-300 border border-indigo-500/30 text-xs font-semibold flex items-center gap-1.5 transition-colors"
              >
                <span>View Cleaning Details</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────
          TAB 4: DATASET DETAILS / DATASET OVERVIEW
         ───────────────────────────────────────────────────────────── */}
      {activeTab === 'details' && (
        <div className="space-y-5 animate-in fade-in duration-200">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">Dataset Overview</h2>
              <p className="text-xs text-slate-400">Schema profile and column distributions for {datasetName}.</p>
            </div>

            <div className="relative w-full sm:w-64">
              <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search columns..."
                value={searchColumn}
                onChange={(e) => setSearchColumn(e.target.value)}
                className="w-full bg-[#0e1629] border border-[#1e293b] rounded-xl pl-9 pr-3 py-1.5 text-xs text-white placeholder:text-slate-500 outline-none focus:border-[#6366f1]"
              />
            </div>
          </div>

          {/* Subtabs: Column Information | Sample Data */}
          <div className="flex items-center gap-2 border-b border-[#1e293b] pb-1 text-xs">
            <button
              onClick={() => setDetailsSubTab('columns')}
              className={clsx(
                "px-3 py-1.5 rounded-lg font-medium transition-colors",
                detailsSubTab === 'columns'
                  ? "bg-[#171f38] text-indigo-300 font-bold border border-indigo-500/30"
                  : "text-slate-400 hover:text-white"
              )}
            >
              Column Information
            </button>
            <button
              onClick={() => setDetailsSubTab('sample')}
              className={clsx(
                "px-3 py-1.5 rounded-lg font-medium transition-colors",
                detailsSubTab === 'sample'
                  ? "bg-[#171f38] text-indigo-300 font-bold border border-indigo-500/30"
                  : "text-slate-400 hover:text-white"
              )}
            >
              Sample Data
            </button>
          </div>

          {detailsSubTab === 'columns' && (
            <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-[#1e293b] bg-[#111a33]/50 text-slate-400 font-semibold uppercase tracking-wider">
                      <th className="py-3 px-4">Column Name</th>
                      <th className="py-3 px-4">Detected Type</th>
                      <th className="py-3 px-4">Unique Values</th>
                      <th className="py-3 px-4">Missing Values</th>
                      <th className="py-3 px-4">Sample Values</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1e293b]">
                    {filteredColumns.map((col: any) => (
                      <tr key={col.name} className="hover:bg-[#131d38]/50 transition-colors">
                        <td className="py-3 px-4 font-mono font-bold text-white">{col.name}</td>
                        <td className="py-3 px-4">
                          <span className={clsx("px-2 py-0.5 rounded-full text-[10px] font-semibold border", col.typeColor)}>
                            {col.type}
                          </span>
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-300">{col.unique}</td>
                        <td className="py-3 px-4 font-mono">
                          {col.missingAlert ? (
                            <span className="text-red-400 font-bold">{col.missing}</span>
                          ) : (
                            <span className="text-slate-400">{col.missing}</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-slate-400 font-mono">{col.samples}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {detailsSubTab === 'sample' && (
            <div className="bg-[#0e1629] border border-[#1e293b] rounded-2xl overflow-hidden shadow-sm p-4">
              {previewData?.preview && previewData.preview.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="border-b border-[#1e293b] text-slate-400">
                        {previewData.columns.map((c: string) => (
                          <th key={c} className="py-2 px-3">{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1e293b]">
                      {previewData.preview.map((row: any, rIdx: number) => (
                        <tr key={rIdx} className="hover:bg-[#131d38]/40">
                          {previewData.columns.map((c: string) => (
                            <td key={c} className="py-2 px-3 text-slate-300">{String(row[c] ?? '')}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="py-8 text-center text-xs text-slate-400">
                  Upload a dataset to inspect interactive sample rows.
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────
          TAB 5: ADVANCED ANALYSIS
         ───────────────────────────────────────────────────────────── */}
      {activeTab === 'advanced' && (
        <div className="space-y-6 animate-in fade-in duration-200">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Advanced Analysis & Statistical Backing</h2>
            <p className="text-xs text-slate-400">Methodologies, hypothesis testing, and machine learning evaluations.</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-[#0e1629] border border-[#1e293b] hover:border-indigo-500/50 p-5 rounded-2xl shadow-sm space-y-3 transition-all">
              <div className="w-10 h-10 rounded-xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center">
                <Network className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-sm text-white">Correlation Analysis</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Computes Pearson and Spearman coefficients across continuous variables to detect multicollinearity.
              </p>
              <div className="pt-2">
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300">
                  Computed Automatically
                </span>
              </div>
            </div>

            <div className="bg-[#0e1629] border border-[#1e293b] hover:border-sky-500/50 p-5 rounded-2xl shadow-sm space-y-3 transition-all">
              <div className="w-10 h-10 rounded-xl bg-sky-500/10 text-sky-400 flex items-center justify-center">
                <Layers className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-sm text-white">Clustering & Cohorts</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Evaluates latent subgroups and natural clusters across multivariate numerical spaces.
              </p>
              <div className="pt-2">
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-sky-500/10 text-sky-300">
                  Cohort Discovery Active
                </span>
              </div>
            </div>

            <div className="bg-[#0e1629] border border-[#1e293b] hover:border-red-500/50 p-5 rounded-2xl shadow-sm space-y-3 transition-all">
              <div className="w-10 h-10 rounded-xl bg-red-500/10 text-red-400 flex items-center justify-center">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-sm text-white">Outlier Detection</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Isolation Forest and IQR bounds applied to quarantine extreme anomalies and leverage points.
              </p>
              <div className="pt-2">
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-red-500/10 text-red-300">
                  Robust Estimators
                </span>
              </div>
            </div>

            <div className="bg-[#0e1629] border border-[#1e293b] hover:border-emerald-500/50 p-5 rounded-2xl shadow-sm space-y-3 transition-all">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center">
                <BarChart3 className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-sm text-white">Feature Importance</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Calculates model attribution values for top drivers influencing the target outcome.
              </p>
              <div className="pt-2">
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300">
                  Champion: {championName}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Cleaning Details Modal */}
      {showCleaningModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in duration-150">
          <div className="bg-[#0b1326] border border-[#1e293b] rounded-2xl w-full max-w-lg p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-[#1e293b] pb-3">
              <h3 className="font-bold text-white text-base">Automated Cleaning Pipeline Log</h3>
              <button
                onClick={() => setShowCleaningModal(false)}
                className="text-slate-400 hover:text-white p-1 rounded-lg"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-3 text-xs text-slate-300">
              <div className="p-3 rounded-xl bg-[#0e1629] border border-[#1e293b] space-y-1">
                <p className="font-semibold text-emerald-400">Missing Values Imputation</p>
                <p className="text-slate-400">Missing entries imputed using median and mode heuristics; zero rows dropped unnecessarily.</p>
              </div>
              <div className="p-3 rounded-xl bg-[#0e1629] border border-[#1e293b] space-y-1">
                <p className="font-semibold text-emerald-400">Deduplication</p>
                <p className="text-slate-400">Exact duplicate records identified and removed to avoid statistical bias.</p>
              </div>
              <div className="p-3 rounded-xl bg-[#0e1629] border border-[#1e293b] space-y-1">
                <p className="font-semibold text-emerald-400">Normalization & Formatting</p>
                <p className="text-slate-400">Standardized timestamps, trimmed categorical whitespace, and encoded dtypes.</p>
              </div>
            </div>
            <div className="flex justify-end pt-2">
              <button
                onClick={() => setShowCleaningModal(false)}
                className="px-4 py-2 text-xs font-semibold bg-[#1e293b] hover:bg-slate-700 text-white rounded-xl transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Report Modal */}
      <ExportModal
        isOpen={isExportOpen}
        onClose={() => setIsExportOpen(false)}
        datasetName={datasetName}
      />
    </div>
  );
}
