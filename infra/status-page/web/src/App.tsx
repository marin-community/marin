import { useAtom } from "jotai";
import { BuildPanel } from "./components/BuildPanel";
import { IrisPanel } from "./components/IrisPanel";
import { NightlyPanel } from "./components/NightlyPanel";
import { ProbesPanel } from "./components/ProbesPanel";
import { autoRefreshAtom } from "./state";

export function App() {
  const [autoRefresh, setAutoRefresh] = useAtom(autoRefreshAtom);

  return (
    <div className="mx-auto max-w-[128rem] px-6 py-8">
      <header className="mb-8 flex items-baseline justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Marin Infra Status</h1>
        <label className="flex items-center gap-2 text-sm text-slate-400">
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
            className="h-4 w-4 accent-emerald-500"
          />
          auto-refresh
        </label>
      </header>

      <div className="space-y-8">
        <NightlyPanel />
        <BuildPanel />
        <IrisPanel />
        <ProbesPanel />
      </div>
    </div>
  );
}
