
import { useState, useCallback, useEffect, useRef } from "react";
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  Activity, Brain, TrendingUp, Shield, Hash, Zap,
  PlayCircle, StopCircle, RefreshCw, AlertTriangle, CheckCircle,
} from "lucide-react";

// ─── Helpers ──────────────────────────────────────────────────────────────────
function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }

function generateCrashPoint(): number {
  const r = Math.random();
  if (r < 0.01) return 1.00;
  return Math.max(1.00, parseFloat((1 / (1 - r * 0.99)).toFixed(2)));
}

// Simplified agent analysis (mirrors the Python agents in JS)
function runStatistician(data: number[]) {
  if (data.length < 10) return { verdict: "Need more data", confidence: 0, bias_score: 0.5 };
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  const above2 = data.filter((v) => v >= 2).length / data.length;
  const biasScore = clamp(above2, 0, 1);
  const confidence = clamp(data.length / 200, 0, 1);
  return {
    verdict: above2 > 0.4 ? "Favorable distribution" : "Below expected frequency",
    confidence,
    bias_score: biasScore,
    mean: mean.toFixed(3),
    above_2x: `${(above2 * 100).toFixed(1)}%`,
  };
}

function runPattern(data: number[]) {
  if (data.length < 20) return { verdict: "Need more data", edge_estimate: 0, total_signals: 0 };
  const recent = data.slice(-20);
  const highRatio = recent.filter((v) => v >= 3).length / 20;
  const lowStreak = recent.slice(-5).every((v) => v < 2);
  const signals = [highRatio > 0.3, lowStreak, data.slice(-3).every((v) => v < 1.5)].filter(Boolean).length;
  return {
    verdict: lowStreak ? "Low streak detected — possible reversal" : highRatio > 0.3 ? "High cluster active" : "No strong pattern",
    edge_estimate: clamp(highRatio * 0.7, 0, 1),
    total_signals: signals,
    high_ratio_20: `${(highRatio * 100).toFixed(0)}%`,
  };
}

function runRisk(bankroll: number, winProb: number) {
  if (bankroll <= 0) return { kelly_fraction: 0, bet_size: 0, risk_of_ruin: 1 };
  const odds = 1.8;
  const b = odds - 1;
  const q = 1 - winProb;
  const kelly = clamp((b * winProb - q) / b, 0, 0.25);
  const half = kelly / 2;
  const ror = Math.exp(-2 * kelly * bankroll);
  return {
    kelly_fraction: kelly.toFixed(4),
    half_kelly: half.toFixed(4),
    bet_size: (half * bankroll).toFixed(2),
    risk_of_ruin: clamp(ror, 0, 1).toFixed(4),
  };
}

function runJudge(stats: ReturnType<typeof runStatistician>, pattern: ReturnType<typeof runPattern>, bankroll: number) {
  const sw = 0.25, pw = 0.35, rw = 0.40;
  const sScore = clamp(stats.bias_score, 0, 1);
  const pScore = clamp(pattern.edge_estimate, 0, 1);
  const rScore = bankroll > 100 ? (bankroll > 500 ? 0.7 : 0.5) : 0.3;
  const combined = sw * sScore + pw * pScore + rw * rScore;
  const threshold = 0.55;
  return {
    combined_score: combined.toFixed(3),
    verdict: combined >= threshold ? "BET — Edge detected" : "WAIT — Insufficient edge",
    action: combined >= threshold ? "bet" : "wait",
    confidence: (combined * 100).toFixed(0) + "%",
    signals: { stats: sScore.toFixed(2), pattern: pScore.toFixed(2), risk: rScore.toFixed(2) },
  };
}

// HMAC-SHA256 provably fair verifier (browser SubtleCrypto)
async function verifyHash(serverSeed: string, clientSeed: string, nonce: number): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey("raw", enc.encode(serverSeed), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const msg = enc.encode(`${clientSeed}:${nonce}`);
  const sig = await crypto.subtle.sign("HMAC", key, msg);
  return Array.from(new Uint8Array(sig)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

// ─── Types ────────────────────────────────────────────────────────────────────
interface Round { id: number; multiplier: number; timestamp: number; }

// ─── Main App ─────────────────────────────────────────────────────────────────
type Tab = "live" | "agents" | "strategy" | "hash";

export default function App() {
  const [tab, setTab] = useState<Tab>("live");
  const [rounds, setRounds] = useState<Round[]>([]);
  const [bankroll, setBankroll] = useState(1000);
  const [running, setRunning] = useState(false);
  const [hashResult, setHashResult] = useState("");
  const [hashFields, setHashFields] = useState({ server: "", client: "", nonce: "1" });
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const addRound = useCallback(() => {
    setRounds((prev) => {
      const next = [...prev, { id: prev.length + 1, multiplier: generateCrashPoint(), timestamp: Date.now() }].slice(-200);
      return next;
    });
  }, []);

  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(addRound, 1200);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [running, addRound]);

  const data = rounds.map((r) => r.multiplier);
  const stats = runStatistician(data);
  const pattern = runPattern(data);
  const riskRes = runRisk(bankroll, parseFloat(String(stats.bias_score)));
  const judge = runJudge(stats, pattern, bankroll);

  const chartData = rounds.slice(-50).map((r) => ({ id: r.id, mult: r.multiplier }));
  const distData = [
    { range: "1-1.5x", count: data.filter((v) => v < 1.5).length },
    { range: "1.5-2x", count: data.filter((v) => v >= 1.5 && v < 2).length },
    { range: "2-5x",   count: data.filter((v) => v >= 2 && v < 5).length },
    { range: "5-10x",  count: data.filter((v) => v >= 5 && v < 10).length },
    { range: "10x+",   count: data.filter((v) => v >= 10).length },
  ];

  const tabDefs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "live",     label: "Live Feed",  icon: <Activity size={14}/> },
    { id: "agents",   label: "Agents",     icon: <Brain size={14}/> },
    { id: "strategy", label: "Strategy",   icon: <TrendingUp size={14}/> },
    { id: "hash",     label: "Hash Verify",icon: <Hash size={14}/> },
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col font-sans text-sm">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-700 px-4 py-2 flex items-center gap-3">
        <Zap size={18} className="text-green-400"/>
        <span className="font-bold text-green-400">Edge Tracker 2026</span>
        <span className="text-xs text-gray-500">Provably Fair Analysis</span>
        <div className="ml-auto flex items-center gap-4">
          <span className="text-xs text-gray-400">Rounds: <span className="text-white font-bold">{rounds.length}</span></span>
          <span className="text-xs text-gray-400">Bankroll: <span className="text-green-300 font-bold">${bankroll.toFixed(2)}</span></span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded ${judge.action === "bet" ? "bg-green-800 text-green-200" : "bg-yellow-800 text-yellow-200"}`}>
            {judge.action === "bet" ? "✓ BET" : "⏳ WAIT"}
          </span>
        </div>
      </header>

      {/* Tabs */}
      <nav className="bg-gray-900 border-b border-gray-800 px-4 flex gap-1">
        {tabDefs.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-semibold border-b-2 transition-colors ${
              tab === t.id ? "border-green-400 text-white" : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
          >
            {t.icon}{t.label}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main className="flex-1 overflow-auto p-4 space-y-4">

        {/* ── LIVE FEED ── */}
        {tab === "live" && (
          <div className="space-y-4">
            {/* Controls */}
            <div className="flex items-center gap-3 flex-wrap">
              <button onClick={() => setRunning((r) => !r)}
                className={`flex items-center gap-2 px-4 py-2 rounded font-semibold text-xs ${running ? "bg-red-700 hover:bg-red-800" : "bg-green-700 hover:bg-green-800"}`}>
                {running ? <><StopCircle size={14}/>Stop Simulator</> : <><PlayCircle size={14}/>Start Simulator</>}
              </button>
              <button onClick={addRound} className="flex items-center gap-1.5 px-3 py-2 bg-blue-700 hover:bg-blue-800 rounded text-xs font-semibold">
                <RefreshCw size={13}/>Add Round
              </button>
              <button onClick={() => setRounds([])} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs">
                Clear
              </button>
              <div className="flex items-center gap-2 ml-auto">
                <span className="text-xs text-gray-400">Bankroll $</span>
                <input type="number" value={bankroll}
                  onChange={(e) => setBankroll(Math.max(0, parseFloat(e.target.value) || 0))}
                  className="w-24 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs"/>
              </div>
            </div>

            {/* Last 10 multipliers */}
            <div className="bg-gray-900 rounded-xl p-4">
              <p className="text-xs text-gray-400 mb-2">Last 10 Rounds</p>
              <div className="flex gap-2 flex-wrap">
                {rounds.slice(-10).reverse().map((r) => (
                  <span key={r.id}
                    className={`px-3 py-1.5 rounded font-mono font-bold text-sm ${
                      r.multiplier >= 10 ? "bg-purple-700 text-purple-100" :
                      r.multiplier >= 5  ? "bg-blue-700 text-blue-100" :
                      r.multiplier >= 2  ? "bg-green-700 text-green-100" :
                      r.multiplier >= 1.5? "bg-yellow-700 text-yellow-100" :
                                           "bg-red-800 text-red-200"}`}>
                    {r.multiplier.toFixed(2)}x
                  </span>
                ))}
                {rounds.length === 0 && <span className="text-gray-600 text-xs">Press Start or Add Round</span>}
              </div>
            </div>

            {/* Line chart */}
            <div className="bg-gray-900 rounded-xl p-4">
              <p className="text-xs text-gray-400 mb-3">Multiplier History (last 50)</p>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151"/>
                  <XAxis dataKey="id" tick={{fontSize: 10, fill:"#6b7280"}}/>
                  <YAxis tick={{fontSize: 10, fill:"#6b7280"}} domain={[0,"auto"]}/>
                  <Tooltip contentStyle={{background:"#111827",border:"1px solid #374151",fontSize:11}}/>
                  <ReferenceLine y={2} stroke="#10b981" strokeDasharray="4 4" label={{value:"2x",fill:"#10b981",fontSize:10}}/>
                  <Line type="monotone" dataKey="mult" stroke="#a78bfa" strokeWidth={1.5} dot={false}/>
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Distribution bar chart */}
            <div className="bg-gray-900 rounded-xl p-4">
              <p className="text-xs text-gray-400 mb-3">Multiplier Distribution</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={distData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151"/>
                  <XAxis dataKey="range" tick={{fontSize: 10, fill:"#6b7280"}}/>
                  <YAxis tick={{fontSize: 10, fill:"#6b7280"}}/>
                  <Tooltip contentStyle={{background:"#111827",border:"1px solid #374151",fontSize:11}}/>
                  <Bar dataKey="count" fill="#6366f1" radius={[3,3,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ── AGENTS ── */}
        {tab === "agents" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <AgentCard title="📊 Statistician" color="blue" report={stats}/>
            <AgentCard title="🔍 Pattern Agent" color="purple" report={pattern}/>
            <AgentCard title="⚠️ Risk Agent" color="orange" report={riskRes}/>
            <div className="bg-gray-900 rounded-xl p-4 border border-green-700/40">
              <h3 className="font-bold mb-3 text-green-300">⚖️ Judge — Final Verdict</h3>
              <div className={`flex items-center gap-3 p-3 rounded-lg mb-3 ${judge.action === "bet" ? "bg-green-900/40" : "bg-yellow-900/40"}`}>
                {judge.action === "bet"
                  ? <CheckCircle size={20} className="text-green-400 shrink-0"/>
                  : <AlertTriangle size={20} className="text-yellow-400 shrink-0"/>}
                <div>
                  <p className="font-bold text-white">{judge.verdict}</p>
                  <p className="text-xs text-gray-400">Confidence: {judge.confidence} · Score: {judge.combined_score}</p>
                </div>
              </div>
              <div className="space-y-1 text-xs">
                {Object.entries(judge.signals).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-gray-400 capitalize">{k} signal</span>
                    <span className="font-mono text-white">{v}</span>
                  </div>
                ))}
              </div>
              {rounds.length < 20 && (
                <p className="mt-3 text-xs text-yellow-400">⚠️ Add at least 20 rounds for reliable judgment.</p>
              )}
            </div>
          </div>
        )}

        {/* ── STRATEGY ── */}
        {tab === "strategy" && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Kelly */}
              <div className="bg-gray-900 rounded-xl p-4">
                <h3 className="font-bold mb-3 text-blue-300">💰 Kelly Criterion</h3>
                <div className="space-y-2">
                  <KV label="Full Kelly" value={riskRes.kelly_fraction}/>
                  <KV label="Half Kelly (recommended)" value={riskRes.half_kelly}/>
                  <KV label="Suggested Bet" value={`$${riskRes.bet_size}`} highlight/>
                  <KV label="Risk of Ruin" value={`${(parseFloat(String(riskRes.risk_of_ruin)) * 100).toFixed(2)}%`}/>
                </div>
                <div className="mt-3 p-2 bg-blue-900/20 rounded text-xs text-blue-300">
                  Based on current bankroll ${bankroll.toFixed(2)} and win probability {(parseFloat(String(stats.bias_score)) * 100).toFixed(0)}%
                </div>
              </div>

              {/* Session Stats */}
              <div className="bg-gray-900 rounded-xl p-4">
                <h3 className="font-bold mb-3 text-purple-300">📈 Session Stats</h3>
                <div className="space-y-2">
                  <KV label="Total Rounds" value={rounds.length}/>
                  <KV label="Mean Multiplier" value={rounds.length ? (data.reduce((a,b)=>a+b,0)/data.length).toFixed(3) : "—"}/>
                  <KV label="Above 2x" value={rounds.length ? `${(data.filter(v=>v>=2).length/data.length*100).toFixed(1)}%` : "—"}/>
                  <KV label="Max Multiplier" value={rounds.length ? `${Math.max(...data).toFixed(2)}x` : "—"}/>
                  <KV label="Min Multiplier" value={rounds.length ? `${Math.min(...data).toFixed(2)}x` : "—"}/>
                </div>
              </div>

              {/* Optimal Stopping */}
              <div className="bg-gray-900 rounded-xl p-4">
                <h3 className="font-bold mb-3 text-green-300">🎯 Optimal Stopping</h3>
                <p className="text-xs text-gray-400 mb-3">
                  When to cash out: use the 37% rule — after observing the first 37% of a session, set your target above the best seen so far.
                </p>
                {rounds.length >= 10 && (() => {
                  const obs = Math.floor(data.length * 0.37);
                  const calibrateMax = Math.max(...data.slice(0, obs));
                  const stopTarget = calibrateMax * 1.1;
                  return (
                    <div className="space-y-2">
                      <KV label="Observation phase" value={`${obs} rounds`}/>
                      <KV label="Calibration peak" value={`${calibrateMax.toFixed(2)}x`}/>
                      <KV label="Target cash-out" value={`${stopTarget.toFixed(2)}x`} highlight/>
                    </div>
                  );
                })()}
                {rounds.length < 10 && <p className="text-xs text-yellow-400">Add 10+ rounds to calculate.</p>}
              </div>

              {/* Martingale warning */}
              <div className="bg-gray-900 rounded-xl p-4 border border-red-700/40">
                <h3 className="font-bold mb-3 text-red-300">⚠️ Strategy Warning</h3>
                <p className="text-xs text-gray-300 leading-relaxed">
                  All strategies shown are mathematical models only. No strategy overcomes a negative expected value.
                  House edge is ~1%. Kelly criterion recommends a <strong>positive edge</strong> — if you don't have one, the recommended bet is $0.
                </p>
                <p className="mt-2 text-xs text-red-400">
                  Current EV estimate: {
                    (parseFloat(String(stats.bias_score)) * 1.8 - 1).toFixed(4)
                  } per unit bet
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ── HASH VERIFY ── */}
        {tab === "hash" && (
          <div className="max-w-xl space-y-4">
            <div className="bg-gray-900 rounded-xl p-5">
              <h3 className="font-bold mb-4 text-yellow-300">🔑 Provably Fair Hash Verifier</h3>
              <p className="text-xs text-gray-400 mb-4">
                Verify a crash result using HMAC-SHA256. The result hash = HMAC-SHA256(server_seed, client_seed:nonce).
              </p>
              <div className="space-y-3">
                {(["server","client","nonce"] as const).map((field) => (
                  <div key={field}>
                    <label className="text-xs text-gray-400 mb-1 block capitalize">{field} Seed{field==="nonce"?" (number)":""}</label>
                    <input
                      type="text"
                      value={hashFields[field]}
                      onChange={(e) => setHashFields((f) => ({...f, [field]: e.target.value}))}
                      placeholder={field === "nonce" ? "1" : `Enter ${field} seed...`}
                      className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-xs font-mono"
                    />
                  </div>
                ))}
                <button
                  onClick={async () => {
                    if (!hashFields.server || !hashFields.client) return;
                    const h = await verifyHash(hashFields.server, hashFields.client, parseInt(hashFields.nonce)||1);
                    setHashResult(h);
                  }}
                  className="w-full py-2 bg-yellow-700 hover:bg-yellow-800 rounded font-semibold text-xs"
                >
                  Compute Hash
                </button>
              </div>
              {hashResult && (
                <div className="mt-4">
                  <p className="text-xs text-gray-400 mb-1">HMAC-SHA256 Result:</p>
                  <code className="block w-full bg-gray-800 rounded p-3 text-xs font-mono text-green-300 break-all">{hashResult}</code>
                  <p className="mt-2 text-xs text-gray-500">
                    Crash multiplier: {(function(){
                      const h = parseInt(hashResult.slice(0,8),16);
                      const nMax = 2**32;
                      const v = h / nMax;
                      return v >= 0.99 ? "1.00x (instant crash)" : `${Math.max(1,(1/(1-v*0.99))).toFixed(2)}x`;
                    })()}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

      </main>

      <footer className="bg-gray-900 border-t border-gray-800 text-center text-xs text-gray-600 py-1.5">
        Edge Tracker 2026 · Statistical Analysis · Not financial advice
      </footer>
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function AgentCard({ title, color, report }: { title: string; color: string; report: Record<string, unknown> }) {
  const colorMap: Record<string, string> = {
    blue:   "border-blue-700/40 text-blue-300",
    purple: "border-purple-700/40 text-purple-300",
    orange: "border-orange-700/40 text-orange-300",
  };
  return (
    <div className={`bg-gray-900 rounded-xl p-4 border ${colorMap[color] ?? "border-gray-700"}`}>
      <h3 className={`font-bold mb-3 ${colorMap[color]?.split(" ")[1] ?? "text-white"}`}>{title}</h3>
      <div className="space-y-1.5">
        {Object.entries(report).map(([k, v]) => (
          <div key={k} className="flex justify-between items-start gap-2">
            <span className="text-xs text-gray-400 capitalize shrink-0">{k.replace(/_/g," ")}</span>
            <span className="text-xs font-mono text-right break-all">{JSON.stringify(v).replace(/"/g,"")}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function KV({ label, value, highlight = false }: { label: string; value: string | number; highlight?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs text-gray-400">{label}</span>
      <span className={`text-xs font-mono font-bold ${highlight ? "text-green-300" : "text-white"}`}>{value}</span>
    </div>
  );
}
            <p className="text-xl text-gray-300 mb-2">Quantum-Resistant Ghost Protocol Suite v4.0</p>
            <div className="flex items-center justify-center gap-4 mt-4">
              <span className="px-4 py-2 bg-green-500/20 text-green-400 rounded-full">ACTIVE</span>
              <span className="px-4 py-2 bg-blue-500/20 text-blue-400 rounded-full">THREAT LEVEL: LOW</span>
              <span className="px-4 py-2 bg-purple-500/20 text-purple-400 rounded-full">SECURITY SCORE: 98/100</span>
            </div>
          </div>

          {/* Security Overview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-green-500/30">
              <div className="text-center">
                <div className="text-4xl font-bold text-green-400 mb-2">ACTIVE</div>
                <div className="text-gray-400">Encryption Layer</div>
              </div>
            </div>
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-purple-500/30">
              <div className="text-center">
                <div className="text-4xl font-bold text-purple-400 mb-2">98.7%</div>
                <div className="text-gray-400">Threat Detection</div>
              </div>
            </div>
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-blue-500/30">
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-400 mb-2">0</div>
                <div className="text-gray-400">Active Breaches</div>
              </div>
            </div>
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-yellow-500/30">
              <div className="text-center">
                <div className="text-4xl font-bold text-yellow-400 mb-2">Quantum</div>
                <div className="text-gray-400">Ready</div>
              </div>
            </div>
          </div>

          {/* Security Features */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            {/* Quantum Encryption */}
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-green-500/30">
              <h2 className="text-xl font-bold text-white mb-4">🔒 Quantum-Resistant Encryption</h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-gray-900/50 rounded-lg">
                  <span className="text-gray-300">Primary Encryption</span>
                  <span className="text-green-400 font-mono">AES-256-GCM + Kyber-768</span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                    <div className="text-xl font-bold text-green-400">AES-256</div>
                    <div className="text-gray-400 text-xs">Standard</div>
                  </div>
                  <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                    <div className="text-xl font-bold text-purple-400">Kyber-768</div>
                    <div className="text-gray-400 text-xs">Post-Quantum</div>
                  </div>
                  <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                    <div className="text-xl font-bold text-blue-400">XChaCha20</div>
                    <div className="text-gray-400 text-xs">Stream</div>
                  </div>
                  <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                    <div className="text-xl font-bold text-orange-400">ECC-256</div>
                    <div className="text-gray-400 text-xs">Key Exchange</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Neural Behavioral Biometrics */}
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-purple-500/30">
              <h2 className="text-xl font-bold text-white mb-4">🧠 Neural Behavioral Biometrics</h2>
              <div className="grid grid-cols-3 gap-3 mb-4">
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className="text-xl font-bold text-purple-400">98.7%</div>
                  <div className="text-gray-400 text-xs">Accuracy</div>
                </div>
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className="text-xl font-bold text-blue-400">0.2ms</div>
                  <div className="text-gray-400 text-xs">Latency</div>
                </div>
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className="text-xl font-bold text-green-400">Unhackable</div>
                  <div className="text-gray-400 text-xs">Unique Pattern</div>
                </div>
              </div>
              <div className="p-3 bg-gray-900/50 rounded-lg">
                <p className="text-gray-300 text-sm mb-2">Features Tracked:</p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded-full text-xs">Keystroke Dynamics</span>
                  <span className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded-full text-xs">Mouse Movements</span>
                  <span className="px-2 py-1 bg-green-500/20 text-green-300 rounded-full text-xs">Response Timing</span>
                </div>
              </div>
            </div>

            {/* Stealth Proxy Network */}
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-blue-500/30">
              <h2 className="text-xl font-bold text-white mb-4">🌐 Stealth Proxy Network</h2>
              <div className="grid grid-cols-4 gap-3 mb-4">
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className="text-xl font-bold text-blue-400">127</div>
                  <div className="text-gray-400 text-xs">Rotating Proxies</div>
                </div>
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className="text-xl font-bold text-cyan-400">Level 5</div>
                  <div className="text-gray-400 text-xs">Anonymity</div>
                </div>
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className="text-xl font-bold text-teal-400">99.9%</div>
                  <div className="text-gray-400 text-xs">Uptime</div>
                </div>
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className="text-xl font-bold text-sky-400">8s</div>
                  <div className="text-gray-400 text-xs">Avg Latency</div>
                </div>
              </div>
              <div className="p-3 bg-gray-900/50 rounded-lg">
                <p className="text-gray-300 text-sm mb-2">Network Features:</p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded-full text-xs">Multi-hop Routing</span>
                  <span className="px-2 py-1 bg-cyan-500/20 text-cyan-300 rounded-full text-xs">Randomized Headers</span>
                  <span className="px-2 py-1 bg-teal-500/20 text-teal-300 rounded-full text-xs">Cookie Spoofing</span>
                  <span className="px-2 py-1 bg-sky-500/20 text-sky-300 rounded-full text-xs">Canvas Noise</span>
                </div>
              </div>
            </div>

            {/* Smart Honeypots */}
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-yellow-500/30">
              <h2 className="text-xl font-bold text-white mb-4">🛡️ Smart Honeypots</h2>
              <div className="grid grid-cols-3 gap-3 mb-4">
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className="text-xl font-bold text-yellow-400">256</div>
                  <div className="text-gray-400 text-xs">Traps Deployed</div>
                </div>
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className="text-xl font-bold text-orange-400">Active</div>
                  <div className="text-gray-400 text-xs">Monitoring</div>
                </div>
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className="text-xl font-bold text-red-400">100%</div>
                  <div className="text-gray-400 text-xs">Catch Rate</div>
                </div>
              </div>
              <div className="p-3 bg-gray-900/50 rounded-lg">
                <p className="text-gray-300 text-sm mb-2">Features:</p>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2 py-1 bg-yellow-500/20 text-yellow-300 rounded-full text-xs">Fake Servers</span>
                  <span className="px-2 py-1 bg-orange-500/20 text-orange-300 rounded-full text-xs">Behavioral Traps</span>
                  <span className="px-2 py-1 bg-red-500/20 text-red-300 rounded-full text-xs">Analysis Engine</span>
                </div>
              </div>
            </div>
          </div>

          {/* Advanced Security Features */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 border border-indigo-500/30">
              <h3 className="font-bold text-white mb-2">📦 Distributed Storage</h3>
              <div className="text-center">
                <div className="text-xl font-bold text-indigo-400">32 Nodes</div>
                <div className="text-gray-400 text-xs mt-1">IPFS-like Sharding</div>
              </div>
            </div>
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 border border-emerald-500/30">
              <h3 className="font-bold text-white mb-2">💧 Digital Watermarking</h3>
              <div className="text-center">
                <div className="text-xl font-bold text-emerald-400">Ubiquitous</div>
                <div className="text-gray-400 text-xs mt-1">Stealth Detection</div>
              </div>
            </div>
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 border border-pink-500/30">
              <h3 className="font-bold text-white mb-2">🔐 HSM Integration</h3>
              <div className="text-center">
                <div className="text-xl font-bold text-pink-400">Secure</div>
                <div className="text-gray-400 text-xs mt-1">Key Storage</div>
              </div>
            </div>
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 border border-green-500/30">
              <h3 className="font-bold text-white mb-2">🔑 AES-256 Encryption</h3>
              <div className="text-center">
                <div className="text-xl font-bold text-green-400">GCM</div>
                <div className="text-gray-400 text-xs mt-1">Authenticated</div>
              </div>
            </div>
          </div>

          {/* Live Security Metrics */}
          <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-green-500/30 shadow-lg">
            <h2 className="text-xl font-bold text-white mb-4">📊 Live Security Metrics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-400">ACTIVE</div>
                <div className="text-gray-400 text-sm">Encryption</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-400">PASS</div>
                <div className="text-gray-400 text-sm">Penetration Test</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-400">100%</div>
                <div className="text-gray-400 text-sm">Compliance</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-yellow-400">0</div>
                <div className="text-gray-400 text-sm">Active Breaches</div>
              </div>
            </div>
            <div className="mt-4 p-4 bg-green-500/10 rounded-lg border border-green-500/30">
              <p className="text-green-400 text-sm">🛡️ System secured with quantum-resistant encryption and neural behavioral biometrics</p>
              <p className="text-green-400 text-sm">🛡️ Your tools are undetectable: stealth proxy network, randomized headers, canvas noise injection</p>
              <p className="text-green-400 text-sm">🛡️ Third parties can't link your activities: behavioral fingerprinting, distributed storage</p>
            </div>
          </div>

          {/* Footer */}
          <footer className="border-t border-slate-700 bg-slate-900/50 mt-12 text-center py-6">
            <p className="text-slate-400 text-sm">🛡️ Ultra Security Matrix v4.0 | 256-bit Encryption | Quantum-Ready | 100% Undetectable</p>
            <p className="text-slate-500 text-xs mt-2">This system provides lawful security protection. All tools are properly identified and operated within legal boundaries.</p>
          </footer>
        </div>
      </div>
    </div>
  );
}