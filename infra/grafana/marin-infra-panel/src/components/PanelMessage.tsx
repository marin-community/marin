import React from 'react';
import { css } from '@emotion/css';
import { useTheme2 } from '@grafana/ui';

interface Props { width: number; height: number; children: React.ReactNode }

/**
 * Centered placeholder for the no-data case. Rendering this instead of throwing
 * keeps a transient empty or errored query from crashing the panel into
 * Grafana's generic error boundary; Grafana still surfaces query errors in its
 * own panel-corner indicator.
 */
export function PanelMessage({ width, height, children }: Props) {
  const theme = useTheme2();
  return (
    <div
      className={css`display:flex;align-items:center;justify-content:center;width:${width}px;height:${height}px;color:${theme.colors.text.secondary};font-size:12px;`}
      aria-label="No data"
    >
      {children}
    </div>
  );
}
