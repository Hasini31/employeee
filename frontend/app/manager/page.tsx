"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler,
} from "chart.js";
import { Bar, Line, Scatter } from "react-chartjs-2";

ChartJS.register(
  CategoryScale, LinearScale, BarElement, PointElement,
  LineElement, Title, Tooltip, Legend, ArcElement, Filler
);

// ─── Types ───────────────────────────────────────────────

interface User { id: number; email: string; name: string; department: string; role: string; }

interface EmployeeSummary {
  employee_id: number; name: string;
  burnout_score: number; burnout_level: string;
  fatigue: number; work_hours: number; mood: string;
  last_updated: string;
  trend: string; trend_arrow: string; trend_insight: string; percentage_change: number;
}

interface Alert {
  employee_id: number; employee_name: string;
  type: string; severity: string; message: string; value: number;
}

interface DayData {
  date: string; avg_burnout: number; avg_fatigue: number;
  high_risk_count: number; submission_count: number;
}

interface WeeklyAnalytics {
  daily_data: DayData[];
  this_week_avg: number; prev_week_avg: number | null;
  burnout_change_pct: number; insights: string[];
}

interface ManagerInsights {
  summary: { total_employees: number; high_risk_employees: number; avg_burnout: number; team_trend: string; team_trend_arrow: string; };
  alerts: Alert[];
  top_risk: EmployeeSummary[];
  employees: EmployeeSummary[];
  weekly_analytics: WeeklyAnalytics;
  correlation_insight: { correlation: number | null; insight: string; };
  team_insights: string[];
  recommendations: string[];
}

interface DrillDownData {
  employee_id: number; name: string; department: string;
  records: Array<{ id: number; mood: string; work_hours: number; fatigue: number; burnout_score: number; burnout_level: string; weekly_trend: string; submission_date: string; created_at: string; sentiment: string; feedback: string; }>;
  trend: { trend: string; trend_arrow: string; percentage_change: number; insight: string; scores: number[]; record_count: number; };
  pattern_summary: string;
}

// ─── Helpers ─────────────────────────────────────────────

function safeDate(raw: string | undefined | null): string {
  if (!raw) return "—";
  try {
    const d = new Date((raw.split("T")[0] || raw) + "T00:00:00");
    if (isNaN(d.getTime())) return raw.slice(0, 10);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch { return raw.slice(0, 10) || "—"; }
}

function RiskBadge({ level }: { level: string }) {
  return <span className={`badge ${level.toLowerCase()}`}>{level}</span>;
}

function TrendArrow({ trend, arrow }: { trend: string; arrow: string }) {
  const cls = { Increasing: "trend-arrow increasing", Decreasing: "trend-arrow decreasing", Stable: "trend-arrow stable", "No Data": "trend-arrow no-data" }[trend] || "trend-arrow no-data";
  return <span className={cls}>{arrow} {trend}</span>;
}

function SeverityDot({ severity }: { severity: string }) {
  return <span className={`severity-dot ${severity}`}></span>;
}

function MiniSparkline({ scores }: { scores: number[] }) {
  if (!scores || scores.length < 2) return null;
  const max = Math.max(...scores), min = Math.min(...scores), range = max - min || 1;
  const w = 100, h = 32, pad = 4;
  const pts = scores.map((s, i) => {
    const x = pad + (i / (scores.length - 1)) * (w - pad * 2);
    const y = h - pad - ((s - min) / range) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const last = scores[scores.length - 1], first = scores[0];
  const stroke = last > first ? "#ef4444" : last < first ? "#10b981" : "#6b7280";
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <polyline fill="none" stroke={stroke} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" points={pts} />
    </svg>
  );
}

// ─── Main Component ───────────────────────────────────────

export default function ManagerDashboard() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [insights, setInsights] = useState<ManagerInsights | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeSection, setActiveSection] = useState<"overview" | "employees" | "alerts" | "analytics">("overview");
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  
  const getInitialDates = () => {
    const today = new Date();
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(today.getDate() - 7);
    return {
      start: sevenDaysAgo.toISOString().split('T')[0],
      end: today.toISOString().split('T')[0]
    };
  };

  const initialDates = getInitialDates();
  const [startDate, setStartDate] = useState(initialDates.start);
  const [endDate, setEndDate] = useState(initialDates.end);
  const [rangeData, setRangeData] = useState<any>(null);

  const fetchInsights = useCallback(async (token: string) => {
    try {
      const res = await fetch("/api/manager-insights", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        if (res.status === 401) {
          localStorage.removeItem("token");
          localStorage.removeItem("user");
          router.push("/manager/login");
          return;
        }
        throw new Error("Failed to fetch manager insights");
      }
      const data = await res.json();
      setInsights(data);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  }, [router]);

  const fetchRangeData = async (start = startDate, end = endDate) => {
    const token = localStorage.getItem("token");
    if (!token) return;
    try {
      const res = await fetch(`/api/analytics-range?startDate=${start}&endDate=${end}&analysisType=daily`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setRangeData(await res.json());
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleApplyRange = () => {
    fetchRangeData(startDate, endDate);
  };

  useEffect(() => {
    fetchRangeData();
  }, []);

  useEffect(() => {
    const token = localStorage.getItem("token");
    const userData = localStorage.getItem("user");
    if (!token || !userData) { router.push("/manager/login"); return; }
    const parsedUser = JSON.parse(userData);
    if (parsedUser.role !== "manager") { router.push("/manager/login"); return; }
    setUser(parsedUser);
    fetchInsights(token);

    // Auto-refresh every 60 seconds
    const interval = setInterval(() => {
      const t = localStorage.getItem("token");
      if (t) fetchInsights(t);
    }, 60000);

    // Refresh on page focus
    const onFocus = () => {
      const t = localStorage.getItem("token");
      if (t) fetchInsights(t);
    };
    window.addEventListener("focus", onFocus);

    return () => { clearInterval(interval); window.removeEventListener("focus", onFocus); };
  }, [router, fetchInsights]);

  const openDrillDown = (employeeId: number) => {
    router.push(`/manager/employee/${employeeId}`);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    router.push("/manager/login");
  };

  // ─── Chart Data ─────────────────────────────────────────

  let plotLabels: string[] = [];
  let plotBurnout: number[] = [];
  let plotFatigue: number[] = [];

  if (rangeData?.daily_data) {
    plotLabels = rangeData.daily_data.map((d: any) => safeDate(d.date));
    plotBurnout = rangeData.daily_data.map((d: any) => d.avg_burnout);
    plotFatigue = rangeData.daily_data.map((d: any) => d.avg_fatigue);
  } else if (!rangeData && insights?.weekly_analytics?.daily_data) {
    plotLabels = insights.weekly_analytics.daily_data.map((d: any) => safeDate(d.date));
    plotBurnout = insights.weekly_analytics.daily_data.map((d: any) => d.avg_burnout);
    plotFatigue = insights.weekly_analytics.daily_data.map((d: any) => d.avg_fatigue);
  }

  const hasData = plotLabels.length > 0;

  const getChartConfig = (label: string, data: number[], color: string, bgColor: string) => ({
    labels: plotLabels,
    datasets: [{
      label,
      data,
      borderColor: color,
      backgroundColor: bgColor,
      fill: true, tension: 0.4, pointRadius: 4,
      pointHoverRadius: 8,
    }]
  });

  const weeklyBurnoutChart = hasData ? getChartConfig("Avg Burnout Score", plotBurnout, "rgb(239, 68, 68)", "rgba(239, 68, 68, 0.1)") : null;
  const weeklyFatigueChart = hasData ? getChartConfig("Avg Fatigue", plotFatigue, "rgb(245, 158, 11)", "rgba(245, 158, 11, 0.1)") : null;

  const chartOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      y: { beginAtZero: true, ticks: { color: "#6b7280" }, grid: { color: "rgba(0,0,0,0.06)" } },
      x: { ticks: { color: "#6b7280" }, grid: { display: false } }
    }
  };



  // ─── Loading / Error ─────────────────────────────────────

  if (loading) return (
    <div className="loading-screen"><span className="spinner"></span><p>Loading dashboard...</p></div>
  );

  if (error) return (
    <>
      <nav className="nav">
        <Link href="/" className="nav-logo">Burnout Detection</Link>
        <div className="nav-links"><button onClick={handleLogout} className="nav-link logout-btn">Logout</button></div>
      </nav>
      <div className="container"><div className="error-alert" style={{ textAlign: "center" }}>{error}</div></div>
    </>
  );

  const s = insights?.summary;
  const criticalAlerts = insights?.alerts.filter(a => a.severity === "critical") || [];
  const warningAlerts = insights?.alerts.filter(a => a.severity === "warning") || [];

  return (
    <>
      <nav className="nav">
        <Link href="/manager" className="nav-logo">Burnout Intelligence</Link>
        <div className="nav-links">
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginRight: "0.75rem" }}>
            Refreshed {lastRefresh.toLocaleTimeString()}
          </span>
          <span className="user-greeting">Manager: {user?.name}</span>
          <button onClick={handleLogout} className="nav-link logout-btn">Logout</button>
        </div>
      </nav>

      <div className="container">
        {/* Header */}
        <div className="dashboard-header fade-in">
          <h1 className="dashboard-title">Burnout <span className="accent">Intelligence</span> Dashboard</h1>
          <p className="dashboard-subtitle">
            Employee-centric wellness monitoring with AI-powered trend analysis and alerts
          </p>
        </div>

        {/* Modern Section Nav */}
        <div style={{ display: "flex", justifyContent: "center", marginBottom: "2rem" }}>
          <div className="modern-dashboard-tabs">
            {(["overview", "employees", "alerts", "analytics"] as const).map(sec => (
              <button key={sec} className={`modern-dashboard-tab ${activeSection === sec ? "active" : ""}`}
                onClick={() => setActiveSection(sec)}>
                <span>{sec.charAt(0).toUpperCase() + sec.slice(1)}</span>
                {sec === "alerts" && insights?.alerts.length ? (
                  <span className="modern-tab-badge">{insights.alerts.length}</span>
                ) : null}
              </button>
            ))}
          </div>
        </div>

        {/* ═══ SECTION: OVERVIEW ═══ */}
        {activeSection === "overview" && (
          <>
            {/* Executive Summary Cards */}
            {/* Modern Executive Summary Cards */}
            <div className="modern-stats-grid slide-up">
              <div className="modern-stat-card">
                <div className="modern-stat-value primary">{s?.total_employees ?? 0}</div>
                <div className="modern-stat-label">Total Employees</div>
              </div>
              <div className="modern-stat-card">
                <div className="modern-stat-value danger-text">{s?.high_risk_employees ?? 0}</div>
                <div className="modern-stat-label">High Risk</div>
              </div>
              <div className="modern-stat-card">
                <div className="modern-stat-value">{s?.avg_burnout ?? 0}</div>
                <div className="modern-stat-label">Avg Burnout Score</div>
              </div>
              <div className="modern-stat-card">
                <div className={`modern-stat-value trend-text ${s?.team_trend === "Increasing" ? "danger-text" : s?.team_trend === "Decreasing" ? "success-text" : ""}`}>
                  <span style={{ marginRight: '4px' }}>{s?.team_trend_arrow}</span>
                  {s?.team_trend}
                </div>
                <div className="modern-stat-label">Team Trend</div>
              </div>
              <div className="modern-stat-card">
                <div className="modern-stat-value warning-text">{criticalAlerts.length}</div>
                <div className="modern-stat-label">Critical Alerts</div>
              </div>
              <div className="modern-stat-card">
                <div className="modern-stat-value">{(insights?.weekly_analytics?.burnout_change_pct ?? 0) > 0 ? "+" : ""}{insights?.weekly_analytics?.burnout_change_pct ?? 0}%</div>
                <div className="modern-stat-label">Week-on-Week Change</div>
              </div>
            </div>

            {/* Team Insights */}
            {insights?.team_insights && insights.team_insights.length > 0 && (
              <div className="card slide-up" style={{ marginBottom: "1.25rem" }}>
                <h3 className="chart-title" style={{ marginBottom: "0.75rem" }}>Team Insights</h3>
                <ul className="insights-list">
                  {insights.team_insights.map((ins, i) => (
                    <li key={i} className="insight-item"><span className="insight-dot"></span>{ins}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Manager Recommendations */}
            {insights?.recommendations && insights.recommendations.length > 0 && (
              <div className="card recommendation-card slide-up" style={{ marginBottom: "1.25rem" }}>
                <h3 className="chart-title" style={{ marginBottom: "0.75rem" }}>Manager Recommendations</h3>
                <ul className="insights-list">
                  {insights.recommendations.map((rec, i) => (
                    <li key={i} className="insight-item recommendation"><span className="insight-icon">→</span>{rec}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Top Risk Employees */}
            <div className="card slide-up" style={{ marginBottom: "1.25rem" }}>
              <h3 className="chart-title" style={{ marginBottom: "1rem" }}>Top 5 Risk Employees</h3>
              {(insights?.top_risk || []).length === 0 ? (
                <div className="empty-state"><p>No high-risk employees</p></div>
              ) : (
                <div className="top-risk-grid">
                  {(insights?.top_risk || []).map((emp) => (
                    <div key={emp.employee_id} className={`top-risk-card ${emp.burnout_level.toLowerCase()}`}
                      onClick={() => openDrillDown(emp.employee_id)} title="Click for drill-down">
                      <div className="top-risk-header">
                        <span className="top-risk-name">{emp.name}</span>
                        <TrendArrow trend={emp.trend} arrow={emp.trend_arrow} />
                      </div>
                      <div className="top-risk-score">{emp.burnout_score}</div>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <RiskBadge level={emp.burnout_level} />
                        <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Fatigue {emp.fatigue}/10</span>
                      </div>
                      <p className="top-risk-insight">{emp.trend_insight}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Correlation Insight */}
            {insights?.correlation_insight?.insight && (
              <div className="card correlation-card slide-up">
                <h3 className="chart-title" style={{ marginBottom: "0.5rem" }}>Correlation Insight</h3>
                <p style={{ color: "var(--text-muted)", lineHeight: 1.6 }}>{insights.correlation_insight.insight}</p>
              </div>
            )}
          </>
        )}

        {/* ═══ SECTION: EMPLOYEES ═══ */}
        {activeSection === "employees" && (
          <div className="card table-card slide-up">
            <h3 className="table-title" style={{ marginBottom: "1rem" }}>Employee Status Table</h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
              Click any row to view 30-day drill-down history.
            </p>
            {(insights?.employees || []).length === 0 ? (
              <div className="empty-state"><p>No employee data yet</p></div>
            ) : (
              <div className="table-responsive">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Employee</th>
                      <th>Latest Score</th>
                      <th>Risk</th>
                      <th>Trend</th>
                      <th>Change %</th>
                      <th>Fatigue</th>
                      <th>Hours</th>
                      <th>Last Updated</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(insights?.employees || []).map((emp) => (
                      <tr key={emp.employee_id}
                        className={`${emp.burnout_level === "High" ? "high-risk" : ""} clickable-row`}
                        onClick={() => { openDrillDown(emp.employee_id); setActiveSection("employees"); }}>
                        <td style={{ fontWeight: 500 }}>{emp.name}</td>
                        <td style={{ fontWeight: 700 }}>{emp.burnout_score}</td>
                        <td><RiskBadge level={emp.burnout_level} /></td>
                        <td><TrendArrow trend={emp.trend} arrow={emp.trend_arrow} /></td>
                        <td className={emp.percentage_change > 0 ? "danger-text" : emp.percentage_change < 0 ? "success-text" : ""}>
                          {emp.percentage_change > 0 ? "+" : ""}{emp.percentage_change}%
                        </td>
                        <td>{emp.fatigue}/10</td>
                        <td>{emp.work_hours}h</td>
                        <td>{safeDate(emp.last_updated)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ═══ SECTION: ALERTS ═══ */}
        {activeSection === "alerts" && (
          <div className="card slide-up">
            <h3 className="table-title" style={{ marginBottom: "1rem" }}>
              Active Alerts ({insights?.alerts.length ?? 0})
            </h3>
            {(insights?.alerts || []).length === 0 ? (
              <div className="empty-state">
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>✓</div>
                <p>No active alerts — team is stable!</p>
              </div>
            ) : (
              <div className="alerts-list">
                {/* Critical */}
                {criticalAlerts.length > 0 && (
                  <div className="alerts-group">
                    <h4 className="alerts-group-title critical">Critical ({criticalAlerts.length})</h4>
                    {criticalAlerts.map((a, i) => (
                      <div key={i} className="alert-item critical">
                        <SeverityDot severity="critical" />
                        <div className="alert-content">
                          <span className="alert-name">{a.employee_name}</span>
                          <span className="alert-message">{a.message}</span>
                        </div>
                        <span className="alert-type-badge">{a.type.replace(/_/g, " ")}</span>
                      </div>
                    ))}
                  </div>
                )}
                {/* Warning */}
                {warningAlerts.length > 0 && (
                  <div className="alerts-group">
                    <h4 className="alerts-group-title warning">Warnings ({warningAlerts.length})</h4>
                    {warningAlerts.map((a, i) => (
                      <div key={i} className="alert-item warning">
                        <SeverityDot severity="warning" />
                        <div className="alert-content">
                          <span className="alert-name">{a.employee_name}</span>
                          <span className="alert-message">{a.message}</span>
                        </div>
                        <span className="alert-type-badge">{a.type.replace(/_/g, " ")}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ═══ SECTION: ANALYTICS ═══ */}
        {activeSection === "analytics" && (
          <>
            <div style={{ textAlign: "center", marginBottom: "1rem", marginTop: "0.25rem", color: "var(--text-muted)", fontSize: "0.9rem" }}>
              Default: previous week analysis. Choose a date range and apply to view custom analytics.
            </div>

            <div className="card slide-up" style={{ padding: "2rem", marginBottom: "2rem", maxWidth: "460px", marginLeft: "auto", marginRight: "auto" }}>
              <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
                <div style={{ flex: 1 }}>
                  <label className="form-label" style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)" }}>START DATE</label>
                  <input type="date" className="form-input" value={startDate} onChange={e => setStartDate(e.target.value)} />
                </div>
                <div style={{ flex: 1 }}>
                  <label className="form-label" style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)" }}>END DATE</label>
                  <input type="date" className="form-input" value={endDate} onChange={e => setEndDate(e.target.value)} />
                </div>
              </div>
              <button className="btn btn-primary" onClick={handleApplyRange} style={{ width: "100%", padding: "0.875rem" }}>
                Apply Range
              </button>
            </div>

            <div className="charts-grid slide-up" style={{ marginBottom: "1.25rem" }}>
              <div className="card chart-card">
                <h3 className="chart-title">Burnout Trend</h3>
                <div className="chart-container">
                  {weeklyBurnoutChart ? <Line data={weeklyBurnoutChart} options={chartOptions} /> : (
                    <div className="empty-state"><p>Not enough data yet</p></div>
                  )}
                </div>
              </div>
              <div className="card chart-card">
                <h3 className="chart-title">Fatigue Trend</h3>
                <div className="chart-container">
                  {weeklyFatigueChart ? <Line data={weeklyFatigueChart} options={chartOptions} /> : (
                    <div className="empty-state"><p>Not enough data yet</p></div>
                  )}
                </div>
              </div>
            </div>

            {insights?.weekly_analytics?.insights && (
              <div className="card slide-up" style={{ marginBottom: "1.25rem" }}>
                <h3 className="chart-title" style={{ marginBottom: "0.75rem" }}>Week-on-Week Insights</h3>
                <ul className="insights-list">
                  {insights.weekly_analytics.insights.map((ins, i) => (
                    <li key={i} className="insight-item"><span className="insight-dot"></span>{ins}</li>
                  ))}
                </ul>
                {insights.weekly_analytics.prev_week_avg != null && (
                  <div style={{ display: "flex", gap: "2rem", marginTop: "1rem" }}>
                    <div className="comparison-stat">
                      <span className="comparison-label">This Week Avg</span>
                      <span className="comparison-value">{insights.weekly_analytics.this_week_avg}</span>
                    </div>
                    <div className="comparison-stat">
                      <span className="comparison-label">Last Week Avg</span>
                      <span className="comparison-value">{insights.weekly_analytics.prev_week_avg}</span>
                    </div>
                    <div className="comparison-stat">
                      <span className="comparison-label">Change</span>
                      <span className={`comparison-value ${insights.weekly_analytics.burnout_change_pct > 0 ? "danger-text" : "success-text"}`}>
                        {insights.weekly_analytics.burnout_change_pct > 0 ? "+" : ""}
                        {insights.weekly_analytics.burnout_change_pct}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* ═══ NO MORE DRILL-DOWN MODAL — REDIRECTS TO PAGE ═══ */}
      </div>
      <style jsx>{`
        /* ──── Modern Dashboard Cards Style ──── */
        .modern-stats-grid {
          display: grid;
          grid-template-columns: repeat(6, 1fr);
          gap: 1.25rem;
          margin-bottom: 2rem;
          font-family: inherit;
        }
        
        @media (max-width: 1200px) {
          .modern-stats-grid { grid-template-columns: repeat(3, 1fr); }
        }
        @media (max-width: 768px) {
          .modern-stats-grid { grid-template-columns: repeat(2, 1fr); }
        }

        .modern-stat-card {
          background: #ffffff;
          border-radius: 12px;
          border: 1px solid #e5e7eb;
          box-shadow: 0 4px 6px rgba(0,0,0,0.02);
          padding: 1.5rem 1rem;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
          height: 140px;
        }

        .modern-stat-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 15px rgba(0,0,0,0.05);
        }

        .modern-stat-value {
          font-size: 2.2rem;
          font-weight: 700;
          color: #111827;
          line-height: 1.1;
          margin-bottom: 0.5rem;
          font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .modern-stat-value.trend-text {
          font-size: 1.6rem; /* Slightly smaller for 'Decreasing' text to fit perfectly */
        }

        .modern-stat-label {
          font-size: 0.85rem;
          font-weight: 500;
          color: #6b7280;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .primary { color: #3b82f6; }
        .danger-text { color: #ef4444; }
        .success-text { color: #10b981; }
        .warning-text { color: #f59e0b; }

        /* ──── Modern Tabs & Phone-Style Notification Badge ──── */
        .modern-dashboard-tabs {
          display: inline-flex;
          background: #f3f4f6;
          border-radius: 9999px;
          padding: 0.35rem;
          gap: 0.25rem;
        }

        .modern-dashboard-tab {
          position: relative;
          padding: 0.6rem 1.25rem;
          font-size: 0.95rem;
          font-weight: 500;
          color: #4b5563;
          background: transparent;
          border: none;
          border-radius: 9999px;
          cursor: pointer;
          transition: all 0.2s ease;
          font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .modern-dashboard-tab:hover {
          color: #111827;
        }

        .modern-dashboard-tab.active {
          background: #ffffff;
          color: #111827;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          font-weight: 600;
        }

        .modern-tab-badge {
          position: absolute;
          top: -3px;
          right: -8px;
          background-color: #ef4444;
          color: white;
          font-size: 0.65rem;
          font-weight: 700;
          width: 18px;
          height: 18px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 50%;
          border: 2px solid #f3f4f6;
          box-shadow: 0 2px 4px rgba(239, 68, 68, 0.4);
          z-index: 10;
        }
        
        .modern-dashboard-tab.active .modern-tab-badge {
          border-color: #ffffff;
        }
      `}</style>
    </>
  );
}
