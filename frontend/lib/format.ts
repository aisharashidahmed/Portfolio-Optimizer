export const percent = (value: number, digits = 1): string =>
  `${(value * 100).toFixed(digits)}%`;

export const signedPercent = (value: number, digits = 1): string =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;

export const ratio = (value: number, digits = 2): string => value.toFixed(digits);

export const money = (value: number, digits = 2): string =>
  value.toLocaleString("en-KE", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

export const compact = (value: number): string =>
  Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(
    value,
  );

/** Human label for a strategy key. */
export const strategyLabel = (key: string): string =>
  ({
    equal_weight: "Equal weight",
    hrp: "HRP",
    mvo: "Mean-variance",
  })[key] ?? key;

/**
 * Colour follows the entity, never its rank.
 *
 * A filter that removes a strategy must not repaint the survivors, so the slot
 * is keyed on the strategy name rather than its index in the current response.
 */
export const SERIES_SLOT: Record<string, string> = {
  equal_weight: "var(--series-1)",
  hrp: "var(--series-2)",
  mvo: "var(--series-3)",
};

export const seriesColour = (key: string, fallbackIndex = 0): string =>
  SERIES_SLOT[key] ?? `var(--series-${(fallbackIndex % 3) + 1})`;
