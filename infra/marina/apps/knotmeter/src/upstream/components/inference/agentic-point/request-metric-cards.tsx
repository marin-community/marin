// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import { useMemo, useState } from "react";

import type { RequestChartData } from "@/hooks/api/use-request-chart-data";
import {
  SegmentedToggle,
  type SegmentedToggleOption,
} from "@/components/ui/segmented-toggle";

import { useLocale } from "@/lib/i18n/use-locale";

import { CHART_SIZES, ChartEmpty, ChartSkeleton } from "./chart-shared";
import { Distribution } from "./distribution";
import { ExpandableChart } from "./expandable-chart";
import { TimeSeriesChart } from "./time-series-chart";
import {
  averageSequenceLengthInFlight,
  rollingRequestMetric,
  timeRollingAverage,
  type RequestMetric,
  type RequestPercentile,
} from "./time-series-math";

const REQUEST_PERCENTILE_OPTIONS: SegmentedToggleOption<RequestPercentile>[] = [
  { value: "p75", label: "P75" },
  { value: "p90", label: "P90" },
];

const LATENCY_METRIC_OPTIONS: SegmentedToggleOption<"ttft" | "e2e">[] = [
  { value: "ttft", label: "TTFT", testId: "latency-metric-ttft" },
  { value: "e2e", label: "E2E", testId: "latency-metric-e2e" },
];

type SequenceMetricView = "distribution" | "inflight";

const REQUEST_CARD_STRINGS = {
  en: {
    distribution: "Distribution",
    inflight: "In-flight avg",
    e2eLatency: "E2E latency",
    interactivity: "Interactivity",
    latencyMetric: "Latency metric",
    points: (count?: number) =>
      count === undefined
        ? "— points"
        : `${count.toLocaleString()} ${count === 1 ? "point" : "points"}`,
    percentile: (metric: string) => `${metric} percentile`,
    overTime: (metric: string) => `${metric} over time`,
    rolling: (pct: string) => `${pct} (rolling 50 req)`,
    cumulative: (pct: string, metric: string) => `Cumulative ${pct} ${metric}`,
    inverseTpot: (pct: string) => `1 / cumulative ${pct} TPOT`,
    inputLength: "Input sequence length",
    outputLength: "Output sequence length",
    distributionTitle: (name: string) => `${name} distribution`,
    avgInflight: (name: string) => `Average ${name} in flight`,
    chartView: (name: string) => `${name} chart view`,
    retrospective:
      "Retrospective: final observed OSL is assigned across each request's lifetime.",
    avgSeries: (name: string) => `Average ${name} in flight (30s avg)`,
    tokensPerRequest: "Tokens / request",
  },
  zh: {
    distribution: "分布",
    inflight: "在途请求平均值",
    e2eLatency: "E2E 延迟",
    interactivity: "交互性",
    latencyMetric: "延迟指标",
    points: (count?: number) =>
      count === undefined
        ? "— 个数据点"
        : `${count.toLocaleString("zh-CN")} 个数据点`,
    percentile: (metric: string) => `${metric} 分位数`,
    overTime: (metric: string) => `${metric} 随时间变化`,
    rolling: (pct: string) => `${pct}（滚动窗口：50 个请求）`,
    cumulative: (pct: string, metric: string) =>
      `截至当前全部请求的 ${pct} ${metric}`,
    inverseTpot: (pct: string) => `截至当前全部请求的 ${pct} TPOT 倒数`,
    inputLength: "输入序列长度",
    outputLength: "输出序列长度",
    distributionTitle: (name: string) => `${name}分布`,
    avgInflight: (name: string) => `在途请求平均 ${name}`,
    chartView: (name: string) => `${name} 图表视图`,
    retrospective:
      "回溯口径：按请求最终观测到的 OSL 计算，并将该值计入请求的整个存续期。",
    avgSeries: (name: string) => `在途请求平均 ${name}（30 秒滑动平均）`,
    tokensPerRequest: "每个请求的 token 数",
  },
} as const;

export function RequestMetricOverTime({
  title,
  metric,
  timeline,
  isLoading,
  latencySelector = false,
}: {
  title: string;
  metric: RequestMetric;
  timeline: RequestChartData | null | undefined;
  isLoading: boolean;
  latencySelector?: boolean;
}) {
  const locale = useLocale();
  const t = REQUEST_CARD_STRINGS[locale];
  const [percentile, setPercentile] = useState<RequestPercentile>("p90");
  const [latencyMetric, setLatencyMetric] = useState<"ttft" | "e2e">("ttft");
  const selectedMetric = latencySelector ? latencyMetric : metric;
  const result = useMemo(
    () =>
      timeline
        ? rollingRequestMetric(
            timeline.requests,
            selectedMetric,
            percentile,
            50,
          )
        : null,
    [timeline, selectedMetric, percentile],
  );
  const metricLabel =
    selectedMetric === "ttft"
      ? "TTFT"
      : selectedMetric === "e2e"
        ? t.e2eLatency
        : t.interactivity;
  const color =
    selectedMetric === "ttft"
      ? "#f59e0b"
      : selectedMetric === "e2e"
        ? "#a855f7"
        : "#06b6d4";
  const pointCount = result?.raw.length;
  const isLatency = selectedMetric !== "interactivity";

  const controls = (
    <div className="flex items-center gap-2">
      {latencySelector && (
        <SegmentedToggle
          value={latencyMetric}
          options={LATENCY_METRIC_OPTIONS}
          onValueChange={(value) => {
            setLatencyMetric(value);
          }}
          ariaLabel={t.latencyMetric}
          testId="latency-metric-toggle"
        />
      )}
      <span
        className="text-xs tabular-nums text-muted-foreground"
        data-testid={`${selectedMetric}-point-count`}
      >
        {t.points(pointCount)}
      </span>
      <SegmentedToggle
        value={percentile}
        options={REQUEST_PERCENTILE_OPTIONS}
        onValueChange={(value) => {
          setPercentile(value);
        }}
        ariaLabel={t.percentile(metricLabel)}
        testId={`${selectedMetric}-percentile-toggle`}
      />
    </div>
  );

  return (
    <ExpandableChart
      title={latencySelector ? t.overTime(metricLabel) : title}
      controls={controls}
      testId={`${metric}-over-time-chart`}
      render={(expanded) => {
        const size = expanded ? CHART_SIZES.expanded : CHART_SIZES.inline;
        if (!timeline) return isLoading ? <ChartSkeleton /> : <ChartEmpty />;
        return (
          <TimeSeriesChart
            series={[
              {
                name: t.rolling(percentile.toUpperCase()),
                data: result?.trend ?? [],
                rawData: result?.raw,
                color,
                strokeWidth: 2.5,
              },
              {
                name: isLatency
                  ? t.cumulative(percentile.toUpperCase(), metricLabel)
                  : t.inverseTpot(percentile.toUpperCase()),
                data: result?.cumulative ?? [],
                color: "#ef4444",
                strokeWidth: 3,
              },
            ]}
            durationS={timeline.durationS}
            yFmt={
              isLatency
                ? (value) =>
                    `${value < 10 ? value.toFixed(1) : value.toFixed(0)}s`
                : (value) => `${value.toFixed(0)}`
            }
            yAxisLabel={
              isLatency
                ? `${metricLabel} (s)`
                : `${t.interactivity} (tok/s/user)`
            }
            {...size}
          />
        );
      }}
    />
  );
}

export function SequenceMetricCard({
  metric,
  timeline,
  timelineLoading,
}: {
  metric: "isl" | "osl";
  timeline: RequestChartData | null | undefined;
  timelineLoading: boolean;
}) {
  const locale = useLocale();
  const t = REQUEST_CARD_STRINGS[locale];
  const [view, setView] = useState<SequenceMetricView>("distribution");
  const acronym = metric.toUpperCase();
  const fullName = metric === "isl" ? t.inputLength : t.outputLength;
  const sequenceOptions: SegmentedToggleOption<SequenceMetricView>[] = [
    { value: "distribution", label: t.distribution },
    { value: "inflight", label: t.inflight },
  ];
  const testPrefix = `${metric}-metric`;
  const values = useMemo(
    () =>
      timeline
        ? timeline.requests
            .map((r) => r[metric])
            .filter(
              (v): v is number => typeof v === "number" && Number.isFinite(v),
            )
        : undefined,
    [timeline, metric],
  );
  const inflightData = useMemo(
    () =>
      timeline
        ? averageSequenceLengthInFlight(timeline.requests, metric)
        : null,
    [timeline, metric],
  );
  return (
    <ExpandableChart
      title={
        view === "distribution"
          ? t.distributionTitle(fullName)
          : t.avgInflight(acronym)
      }
      testId={`${testPrefix}-chart`}
      controls={
        <SegmentedToggle
          value={view}
          options={sequenceOptions.map((option) => ({
            ...option,
            testId: `${testPrefix}-${option.value}`,
          }))}
          onValueChange={(value) => {
            setView(value);
          }}
          ariaLabel={t.chartView(acronym)}
          testId={`${testPrefix}-toggle`}
          buttonClassName="px-2 py-1 text-xs"
        />
      }
      render={(expanded) => {
        const size = expanded ? CHART_SIZES.expanded : CHART_SIZES.inline;
        if (view === "distribution") {
          if (values && values.length > 0)
            return (
              <Distribution
                values={values}
                unit={locale === "zh" ? "token" : "tokens"}
                {...size}
              />
            );
          return timelineLoading ? <ChartSkeleton /> : <ChartEmpty />;
        }
        if (!timeline)
          return timelineLoading ? <ChartSkeleton /> : <ChartEmpty />;
        const raw = inflightData ?? [];
        return (
          <div>
            {metric === "osl" && (
              <p className="mb-2 text-xs text-muted-foreground">
                {t.retrospective}
              </p>
            )}
            <TimeSeriesChart
              series={[
                {
                  name: t.avgSeries(acronym),
                  data: timeRollingAverage(raw, 30),
                  rawData: raw,
                  color: metric === "isl" ? "#3b82f6" : "#a855f7",
                  strokeWidth: 2.5,
                },
              ]}
              durationS={timeline.durationS}
              yAxisLabel={t.tokensPerRequest}
              {...size}
            />
          </div>
        );
      }}
    />
  );
}
