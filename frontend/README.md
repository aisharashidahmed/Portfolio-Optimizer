# Frontend

Next.js App Router + TypeScript. No UI framework, no chart library.

```bash
npm install
npm run dev          # http://localhost:3000
npm test             # 20 tests, no network
npm run typecheck
npm run build
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
  api.ts             typed client; clears the token on an authenticated 401
  auth.ts            token store + subscription
  format.ts          formatters + the fixed series-colour map
```

## Auth

`lib/auth.ts` owns the token and notifies subscribers when it changes, so a 401
anywhere drops every mounted component back to the sign-in card without a
context provider.

This matters because a stored token can stop working while remaining perfectly
well-formed: it expires, or the account it names is deleted (a reset database).
Both return an identical 401 — deliberately, so the endpoint cannot be used to
probe which accounts exist. Without the self-heal the UI renders that error
forever and the only escape is clearing `localStorage` by hand.

The clear is guarded on whether the request was authenticated, because
`/auth/login` also answers 401 for a wrong password and signing the user out
mid-login attempt would be nonsense.

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
