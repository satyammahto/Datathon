import { useEffect } from "react";
import { Sidebar } from './Sidebar';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useStore } from '../store/useStore';
import { fetchWithAuth, API_BASE_URL } from '../utils/apiClient';

export function Layout() {
  const user = useStore(state => state.user);
  const token = useStore(state => state.token);
  const setUser = useStore(state => state.setUser);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (token && !user) {
      fetchWithAuth(`${API_BASE_URL}/api/users/me`)
        .then(res => {
          if (res.ok) return res.json();
          throw new Error('Failed to fetch user');
        })
        .then(data => {
          setUser(data);
        })
        .catch(err => console.error('Layout user fetch error:', err));
    }
  }, [token, user, setUser]);

  // Display user matches the reference image: "Satyam Mahto"
  const displayName = user?.username || 'Satyam Mahto';
  const initial = displayName.charAt(0).toUpperCase();

  const dashboardContract = useStore(state => state.dashboardContract);
  const activeDatasetName = dashboardContract?.dataset_name || "Customer_Sales_Data";

  // Dynamic breadcrumb matching the screenshot
  const getBreadcrumb = () => {
    const path = location.pathname;
    if (path === '/' || path.startsWith('/dashboard')) {
      return (
        <div className="flex items-center gap-2 text-xs font-medium text-slate-400">
          <span
            onClick={() => navigate('/reports')}
            className="hover:text-indigo-400 cursor-pointer transition-colors"
          >
            Reports
          </span>
          <span className="text-slate-600">&gt;</span>
          <span className="text-slate-200 font-semibold">{activeDatasetName}</span>
        </div>
      );
    }
    if (path === '/reports') {
      return (
        <div className="flex items-center gap-2 text-xs font-medium text-slate-400">
          <span className="text-slate-200 font-semibold">Reports</span>
        </div>
      );
    }
    if (path === '/new-analysis') {
      return (
        <div className="flex items-center gap-2 text-xs font-medium text-slate-400">
          <span
            onClick={() => navigate('/reports')}
            className="hover:text-indigo-400 cursor-pointer transition-colors"
          >
            Reports
          </span>
          <span className="text-slate-600">&gt;</span>
          <span className="text-slate-200 font-semibold">New Analysis</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2 text-xs font-medium text-slate-400">
        <span className="text-slate-200 font-semibold capitalize">{path.replace('/', '')}</span>
      </div>
    );
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#0b1326] text-slate-100 font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#0b1326]">
        {/* Top Header */}
        <header className="h-14 border-b border-[#1e293b] flex items-center justify-between px-6 bg-[#0e1629]/60 backdrop-blur-md z-10">
          <div>
            {getBreadcrumb()}
          </div>
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-full bg-[#3b82f6] text-white font-bold text-xs flex items-center justify-center shadow-sm">
              {initial}
            </div>
            <span className="text-xs font-semibold text-slate-300">
              {displayName}
            </span>
          </div>
        </header>

        {/* Main Workspace Area */}
        <main className="flex-1 overflow-y-auto p-6 relative bg-[#0b1326]">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
