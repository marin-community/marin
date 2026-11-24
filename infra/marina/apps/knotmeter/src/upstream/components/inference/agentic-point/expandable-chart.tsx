// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import { useState, type ReactNode } from "react";
import { Maximize2 } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import { useLocale } from "@/lib/i18n/use-locale";

export function ExpandableChart({
  title,
  render,
  controls,
  testId,
}: {
  title: string;
  render: (expanded: boolean) => ReactNode;
  controls?: ReactNode;
  testId?: string;
}) {
  const [open, setOpen] = useState(false);
  const locale = useLocale();

  return (
    <div
      className="rounded-lg border border-border/40 bg-card/40 p-4"
      data-testid={testId}
    >
      <div className="flex items-start justify-between mb-3 gap-2">
        <h2 className="text-sm font-semibold text-foreground">{title}</h2>
        <div className="flex items-center gap-2">
          {controls}
          <button
            type="button"
            aria-label={locale === "zh" ? "展开图表" : "Expand chart"}
            onClick={() => {
              setOpen(true);
            }}
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <Maximize2 className="size-4" />
          </button>
        </div>
      </div>
      {render(false)}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-[min(96vw,1400px)] w-[min(96vw,1400px)]">
          <DialogHeader>
            <div className="flex items-center justify-between gap-3 pr-8">
              <DialogTitle>{title}</DialogTitle>
              {controls}
            </div>
          </DialogHeader>
          <div className="w-full">{render(true)}</div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
