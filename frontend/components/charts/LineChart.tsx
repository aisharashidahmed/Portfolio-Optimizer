"use client";

import { useCallback, useId, useMemo, useRef, useState } from "react";

export type Point = { date: string; value: number };
export type Series = { key: string; label: string; colour: string; points: Point[] };

type Props = {
  series: Series[];
  height?: number;
  /** Formats the y axis and tooltip values. */
  format: (value: number) => string;
  /** Drawn as a hairline, e.g. 1.0 on an equity curve. */
  baseline?: number;
  yLabel?: string;
};

const PAD = { top: 12, right: 16, bottom: 28, left: 56 };

/**
 * Multi-series line chart with a crosshair and shared tooltip.
 *
 * Hand-rolled SVG rather than a chart library: the mark specs this project
 * follows (2px strokes, a 2px surface ring on overlapping markers, selective
 * direct labels, hairline grid) are easier to hit exactly than to configure.
 *
 * One y axis, always. Two measures of different scale get two charts.
 */
export default function LineChart({
  series,
  height = 260,
  format,
  baseline,
  yLabel,
}: Props) {
  const clipId = useId();
  const svgRef = useRef<SVGSVGElement>(null);
  const observerRef = useRef<ResizeObserver | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [width, setWidth] = useState(720);

  /**
   * Measure the container and keep measuring as it resizes.
   *
   * A callback ref rather than an effect: it binds to whichever branch mounts
   * (the empty state or the chart), and re-binds automatically when one
   * replaces the other. A plain ref would only measure on render, leaving the
   * SVG viewBox stuck at its first width when the window changes size.
   */
  const setContainer = useCallback((node: HTMLDivElement | null) => {
    observerRef.current?.disconnect();
    observerRef.current = null;
    if (!node) return;

    const measure = () => {
      const next = node.clientWidth;
      if (next > 0) {
        setWidth((current) => (Math.abs(current - next) > 1 ? next : current));
      }
    };

    measure();

    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    observerRef.current = observer;
  }, []);

  const populated = series.filter((s) => s.points.length > 0);

  const geometry = useMemo(() => {
    if (populated.length === 0) return null;

    const length = Math.max(...populated.map((s) => s.points.length));
    const values = populated.flatMap((s) => s.points.map((p) => p.value));
    if (baseline !== undefined) values.push(baseline);

    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
      min -= 0.5;
      max += 0.5;
    }
    const headroom = (max - min) * 0.08;
    min -= headroom;
    max += headroom;

    const plotWidth = Math.max(1, width - PAD.left - PAD.right);
    const plotHeight = Math.max(1, height - PAD.top - PAD.bottom);

    const x = (i: number) =>
      PAD.left + (length <= 1 ? plotWidth / 2 : (i / (length - 1)) * plotWidth);
    const y = (v: number) =>
      PAD.top + plotHeight - ((v - min) / (max - min)) * plotHeight;

    const ticks = 4;
    const yTicks = Array.from({ length: ticks + 1 }, (_, i) => min + ((max - min) * i) / ticks);

    return { length, x, y, min, max, plotWidth, plotHeight, yTicks };
  }, [populated, width, height, baseline]);

  // The container always mounts, even with no data, so the ResizeObserver is
  // bound before the first series arrives.
  if (!geometry) {
    return (
      <div ref={setContainer} style={{ width: "100%" }}>
        <p className="muted small" style={{ padding: "24px 0" }}>
          No data to plot.
        </p>
      </div>
    );
  }

  const { length, x, y, yTicks, plotHeight } = geometry;

  const longest = populated.reduce((a, b) =>
    a.points.length >= b.points.length ? a : b,
  );

  const handleMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const scale = width / rect.width;
    const localX = (event.clientX - rect.left) * scale;
    const ratio = (localX - PAD.left) / Math.max(1, width - PAD.left - PAD.right);
    const index = Math.round(ratio * (length - 1));
    setHoverIndex(Math.min(length - 1, Math.max(0, index)));
  };

  const hovered = hoverIndex !== null ? longest.points[hoverIndex] : null;

  return (
    <div ref={setContainer} style={{ width: "100%" }}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height={height}
        role="img"
        aria-label={`Line chart${yLabel ? ` of ${yLabel}` : ""} with ${populated.length} series. The table below lists the same values.`}
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
        style={{ display: "block", overflow: "visible" }}
      >
        <defs>
          <clipPath id={clipId}>
            <rect
              x={PAD.left}
              y={PAD.top}
              width={width - PAD.left - PAD.right}
              height={plotHeight}
            />
          </clipPath>
        </defs>

        {/* Solid hairline grid, one shade off the surface. Never dashed. */}
        {yTicks.map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              x2={width - PAD.right}
              y1={y(tick)}
              y2={y(tick)}
              stroke="var(--gridline)"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 8}
              y={y(tick)}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize={11}
              fill="var(--text-muted)"
              style={{ fontVariantNumeric: "tabular-nums" }}
            >
              {format(tick)}
            </text>
          </g>
        ))}

        {baseline !== undefined && (
          <line
            x1={PAD.left}
            x2={width - PAD.right}
            y1={y(baseline)}
            y2={y(baseline)}
            stroke="var(--axis)"
            strokeWidth={1}
          />
        )}

        {hoverIndex !== null && (
          <line
            x1={x(hoverIndex)}
            x2={x(hoverIndex)}
            y1={PAD.top}
            y2={PAD.top + plotHeight}
            stroke="var(--axis)"
            strokeWidth={1}
          />
        )}

        <g clipPath={`url(#${clipId})`}>
          {populated.map((s) => (
            <path
              key={s.key}
              d={s.points
                .map((p, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(p.value)}`)
                .join(" ")}
              fill="none"
              stroke={s.colour}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
        </g>

        {/* Selective direct labels: the endpoint only, never every point. */}
        {populated.map((s) => {
          const last = s.points[s.points.length - 1];
          if (!last) return null;
          return (
            <circle
              key={`${s.key}-end`}
              cx={x(s.points.length - 1)}
              cy={y(last.value)}
              r={3.5}
              fill={s.colour}
              stroke="var(--surface-1)"
              strokeWidth={2}
            />
          );
        })}

        {hoverIndex !== null &&
          populated.map((s) => {
            const point = s.points[Math.min(hoverIndex, s.points.length - 1)];
            if (!point) return null;
            return (
              <circle
                key={`${s.key}-hover`}
                cx={x(hoverIndex)}
                cy={y(point.value)}
                r={4}
                fill={s.colour}
                stroke="var(--surface-1)"
                strokeWidth={2}
              />
            );
          })}

        <line
          x1={PAD.left}
          x2={width - PAD.right}
          y1={PAD.top + plotHeight}
          y2={PAD.top + plotHeight}
          stroke="var(--axis)"
          strokeWidth={1}
        />

        {[0, Math.floor((length - 1) / 2), length - 1].map((i, n) => (
          <text
            key={`${i}-${n}`}
            x={x(i)}
            y={height - 8}
            textAnchor={n === 0 ? "start" : n === 2 ? "end" : "middle"}
            fontSize={11}
            fill="var(--text-muted)"
          >
            {longest.points[i]?.date.slice(0, 7)}
          </text>
        ))}
      </svg>

      <div
        aria-live="polite"
        style={{
          minHeight: 26,
          marginTop: 6,
          fontSize: "0.8rem",
          color: "var(--text-secondary)",
          display: "flex",
          gap: 14,
          flexWrap: "wrap",
        }}
      >
        {hovered && (
          <>
            <strong style={{ color: "var(--text-primary)", fontWeight: 500 }}>
              {hovered.date}
            </strong>
            {populated.map((s) => {
              const point = s.points[Math.min(hoverIndex!, s.points.length - 1)];
              if (!point) return null;
              return (
                <span key={s.key}>
                  <span className="swatch" style={{ background: s.colour }} />
                  {s.label} {format(point.value)}
                </span>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
}
