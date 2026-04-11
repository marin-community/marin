import { useAtom } from "jotai";
import { FerryPanel } from "./components/FerryPanel";
import { OrchPanel } from "./components/OrchPanel";
import { autoRefreshAtom } from "./state";

export function App() {
  const [autoRefresh, setAutoRefresh] = useAtom(autoRefreshAtom);

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <header className="mb-8 flex items-baseline justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Marin status</h1>
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
        <FerryPanel />
        <OrchPanel />
      </div>
    </div>
  );
}
