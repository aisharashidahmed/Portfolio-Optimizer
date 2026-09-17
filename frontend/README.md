# Frontend

Next.js App Router + TypeScript. No UI framework, no chart library.

```bash
npm install
npm run dev          # http://localhost:3000
npm run build
npx tsc --noEmit     # typecheck
```

Expects the API on `http://localhost:8000`. Override with
`NEXT_PUBLIC_API_BASE` (see `.env.example`).

## Layout

```
app/
  page.tsx           currency risk — the landing page
  optimize/          allocation + covariance diagnostics
  backtest/          walk-forward comparison
  nse/               NSE ingestion coverage (public, no sign-in)
components/
  charts/            hand-written SVG: LineChart, BarChart
  TableView.tsx      the accessible twin of every chart
  AuthGate.tsx       demo sign-in
lib/
  api.ts             typed client, token handling
  format.ts          formatters + the fixed series-colour map
```

## Charts

Hand-written SVG rather than a library, because the mark specifications are
easier to hit exactly than to configure: 2px strokes, a 2px surface ring on
overlapping markers, solid hairline gridlines, direct labels only at endpoints.

Three rules the charts hold to:

- **The palette is validated, not eyeballed.** The three series slots clear an
  all-pairs colour-vision-deficiency floor (worst ΔE 9.2 light / 9.4 dark)
  against the exact surfaces used. Aqua measures 2.74:1 on the light surface,
  below the 3:1 bar, so charts using it carry direct labels and a table view as
  relief rather than relying on colour.
- **Colour follows the entity, never its rank.** Strategies are pinned to slots
  in `lib/format.ts`, so filtering one out never repaints the survivors.
- **Every chart has a table twin.** A tooltip enhances a value; it never gates
  one.

No dual-axis charts. Two measures of different scale get two charts.

## Theming

Light and dark are both selected, not an automatic inversion. Tokens live in
`app/globals.css`: the light palette on bare `:root`, dark redefined under both
`@media (prefers-color-scheme: dark)` (guarded so an explicit light choice wins)
and `:root[data-theme="dark"]`.
