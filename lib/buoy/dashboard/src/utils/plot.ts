// plotly.js-dist-min is a large UMD bundle with no bundled types (declared `any`
// in env.d.ts). Import it once here so every chart shares the single instance.
import Plotly from 'plotly.js-dist-min'

export default Plotly

// scrollZoom: trackpad-pinch / wheel zoom; dragmode 'pan' pans; double-click autoscales.
export const CHART_CONFIG = {
  responsive: true,
  displaylogo: false,
  scrollZoom: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
}

export const LINE_COLOR = '#1f77b4' // one blue for every chart
