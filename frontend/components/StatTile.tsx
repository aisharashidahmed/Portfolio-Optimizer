type Props = {
  label: string;
  value: string;
  sub?: string;
  /** Tints the value for a directional number. Omit for neutral figures. */
  tone?: "neutral" | "up" | "down";
};

/**
 * A single number is a stat tile, not a one-bar chart.
 */
export default function StatTile({ label, value, sub, tone = "neutral" }: Props) {
  const colour =
    tone === "up"
      ? "var(--delta-up)"
      : tone === "down"
        ? "var(--delta-down)"
        : "var(--text-primary)";

  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value" style={{ color: colour }}>
        {value}
      </div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}
