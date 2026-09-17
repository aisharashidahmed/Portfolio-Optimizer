/** Typed client for the FastAPI backend. */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

const TOKEN_KEY = "portfolio.token";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** localStorage can throw in a private window; never let that break a render. */
export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token === null) localStorage.removeItem(TOKEN_KEY);
    else localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* ignore */
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  auth = true,
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");

  if (auth) {
    const token = getToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(
      `Cannot reach the API at ${API_BASE}. Start it with: uvicorn src.api.main:app --reload`,
      0,
    );
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        // FastAPI validation errors arrive as a list of field problems.
        detail = body.detail
          .map((d: { loc?: string[]; msg?: string }) => {
            const field = d.loc?.filter((p) => p !== "body").join(".");
            return field ? `${field}: ${d.msg}` : d.msg;
          })
          .join("; ");
      }
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(detail, response.status);
  }

  return response.json() as Promise<T>;
}

/* ---------------------------------------------------------------- types -- */

export type Metrics = {
  expected_annual_return: number;
  annual_volatility: number;
  sharpe_ratio: number;
};

export type OptimizeResponse = {
  tickers: string[];
  weights: number[];
  algorithm: string;
  metrics: Metrics;
  missing_tickers: string[];
  covariance_diagnostics: {
    estimator: string;
    shrinkage_intensity: number;
    condition_number: number;
    n_observations: number;
    n_assets: number;
    observations_per_asset: number;
  };
};

export type CurvePoint = { date: string; value: number };

export type Performance = {
  total_return: number;
  annualised_return: number;
  annualised_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  value_at_risk_95: number;
  conditional_value_at_risk_95: number;
  win_rate: number;
  n_periods: number;
};

export type StrategyResult = {
  strategy: string;
  curve: CurvePoint[];
  performance: Partial<Performance>;
  final_weights: Record<string, number>;
  average_turnover: number;
  total_cost_drag: number;
  error: string | null;
};

export type BacktestResponse = {
  tickers: string[];
  start_date: string;
  end_date: string;
  transaction_cost_bps: number;
  strategies: StrategyResult[];
};

export type CurrencyAsset = {
  ticker: string;
  asset_return_usd: number;
  currency_return: number;
  interaction: number;
  total_return_kes: number;
  asset_volatility: number;
  currency_volatility: number;
  total_volatility_kes: number;
  correlation: number;
  currency_share_of_variance: number;
  diversification_effect: number;
  n_periods: number;
};

export type CurrencyResponse = {
  start_date: string;
  end_date: string;
  fx: {
    start_rate: number;
    end_rate: number;
    total_depreciation: number;
    curve: CurvePoint[];
  };
  assets: CurrencyAsset[];
};

export type NseResponse = {
  tickers_known: number;
  price_rows: number;
  latest_trade_date: string | null;
  usable_for_covariance: number;
  listings: { ticker: string; close_kes: number; trade_date: string }[];
};

/* ------------------------------------------------------------- requests -- */

export const api = {
  signup: (email: string, password: string) =>
    request<{ access_token: string }>(
      "/auth/signup",
      { method: "POST", body: JSON.stringify({ email, password }) },
      false,
    ),

  login: (email: string, password: string) =>
    request<{ access_token: string }>(
      "/auth/login",
      { method: "POST", body: JSON.stringify({ email, password }) },
      false,
    ),

  optimize: (body: {
    tickers: string[];
    start_date: string;
    end_date: string;
    algorithm: string;
    max_weight: number;
    covariance_estimator: string;
  }) =>
    request<OptimizeResponse>("/optimize/sync", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  backtest: (body: {
    tickers: string[];
    start_date: string;
    end_date: string;
    algorithm: string;
    max_weight: number;
    training_days: number;
    rebalance_every: number;
    transaction_cost_bps: number;
  }) =>
    request<BacktestResponse>("/backtest", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  currency: (body: { tickers: string[]; start_date: string; end_date: string }) =>
    request<CurrencyResponse>("/currency", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  nse: () => request<NseResponse>("/nse", {}, false),
};
