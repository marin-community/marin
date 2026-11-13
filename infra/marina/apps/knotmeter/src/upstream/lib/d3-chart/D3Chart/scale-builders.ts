// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import * as d3 from "d3";

import type { ContinuousScale } from "../types";
import type { ScaleConfig } from "./types";

export type BuiltScale =
  | d3.ScaleBand<string>
  | d3.ScaleLinear<number, number>
  | d3.ScaleLogarithmic<number, number>
  | d3.ScaleTime<number, number>;

export function buildScale(
  config: ScaleConfig,
  range: [number, number],
): BuiltScale {
  switch (config.type) {
    case "band": {
      return d3
        .scaleBand<string>()
        .domain(config.domain)
        .range(range)
        .padding(config.padding ?? 0.1);
    }

    case "linear": {
      const s = d3.scaleLinear().domain(config.domain).range(range);
      return config.nice === false ? s : s.nice();
    }

    case "log": {
      const l = d3.scaleLog().domain(config.domain).range(range);
      return config.nice === false ? l : l.nice();
    }

    case "time": {
      const t = d3.scaleTime().domain(config.domain).range(range);
      return config.nice === false ? t : t.nice();
    }
  }
}

export function isContinuousScale(scale: BuiltScale): scale is ContinuousScale {
  return "invert" in scale && !("bandwidth" in scale);
}

export function isBandScale(scale: BuiltScale): scale is d3.ScaleBand<string> {
  return "bandwidth" in scale;
}
