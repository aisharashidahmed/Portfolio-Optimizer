"use client";

import { useState } from "react";

export type Bar = { label: string; value: number };

type Props = {
  bars: Bar[];
  format: (value: number) => string;
  /**
   * One series, one colour. Never a value ramp across nominal categories --
   * that double-encodes bar length as hue and burns the only free channel.
   */
  colour?: string;
  maxValue?: number;
};

const ROW_HEIGHT = 26;
const LABEL_WIDTH = 78;

/**
 * Horizontal bar chart for allocations.
 *
 * Horizontal because ticker labels read left-to-right without rotation, and
 * because an allocation list is naturally ranked.
 */
export default function BarChart({
  bars,
  format,
  colour = "var(--series-1)",
  maxValue,
}: Props) {
  const [hovered, setHovered] = useState<string | null>(null);

  if (bars.length === 0) {
    return (
      <p className="muted small" style={{ padding: "16px 0" }}>
        No allocation to show.
      </p>
    );
  }

  const max = maxValue ?? Math.max(...bars.map((b) => b.value), 0.0001);

  return (
    <div style={{ width: "100%" }}>
      {bars.map((bar) => {
        const fraction = Math.max(0, bar.value / max);
        const isHovered = hovered === bar.label;

        return (
          <div
            key={bar.label}
            onMouseEnter={() => setHovered(bar.label)}
            onMouseLeave={() => setHovered(null)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              /* Hit target comfortably exceeds the mark itself. */
              height: ROW_HEIGHT,
              cursor: "default",
            }}
          >
            <span
              style={{
                width: LABEL_WIDTH,
                flexShrink: 0,
                fontSize: "0.8rem",
                color: "var(--text-secondary)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
              title={bar.label}
            >
              {bar.label}
            </span>

            <div
              style={{
                flex: 1,
                height: 12,
                minWidth: 0,
                background: "var(--gridline)",
                borderRadius: 3,
                position: "relative",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${fraction * 100}%`,
                  height: "100%",
                  background: colour,
                  /* 4px rounded data-end, anchored square to the baseline. */
                  borderRadius: "0 4px 4px 0",
                  opacity: isHovered ? 0.85 : 1,
                  transition: "width 200ms ease, opacity 120ms ease",
                }}
              />
            </div>

            {/* Direct label on every bar: the list is short, and aqua-on-light
                needs relief from the contrast warning. */}
            <span
              style={{
                width: 60,
                flexShrink: 0,
                textAlign: "right",
                fontSize: "0.8rem",
                fontVariantNumeric: "tabular-nums",
                color: "var(--text-primary)",
              }}
            >
              {format(bar.value)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
