"use client";

import { useId, useState } from "react";

type Props = {
  caption: string;
  columns: string[];
  rows: (string | number)[][];
  /** Colour chips in the first column, keyed by row index. */
  swatches?: (string | undefined)[];
};

/**
 * The accessible twin of a chart.
 *
 * Every chart in this app ships one. A tooltip must enhance a value, never be
 * the only way to read it, and three of the light-mode series colours sit below
 * 3:1 against the surface -- so the table is the relief, not a nicety.
 */
export default function TableView({ caption, columns, rows, swatches }: Props) {
  const id = useId();
  const [open, setOpen] = useState(false);

  return (
    <div style={{ marginTop: 12 }}>
      <button
        type="button"
        className="btn btn-secondary"
        style={{ fontSize: "0.78rem", padding: "5px 10px" }}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((value) => !value)}
      >
        {open ? "Hide" : "Show"} data table
      </button>

      <div id={id} hidden={!open} className="table-wrap">
        <table>
          <caption
            style={{
              captionSide: "top",
              textAlign: "left",
              fontSize: "0.78rem",
              color: "var(--text-secondary)",
              paddingBottom: 6,
            }}
          >
            {caption}
          </caption>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column} scope="col">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex}>
                    {cellIndex === 0 && swatches?.[rowIndex] && (
                      <span
                        className="swatch"
                        style={{ background: swatches[rowIndex] }}
                      />
                    )}
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
