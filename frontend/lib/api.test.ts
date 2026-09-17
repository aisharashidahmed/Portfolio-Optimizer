import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { getToken, resetTokenListeners, setToken } from "./auth";

function mockResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

describe("401 handling", () => {
  beforeEach(() => {
    localStorage.clear();
    resetTokenListeners();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("clears a stored token when an authenticated request is rejected", async () => {
    /* The regression this exists for: a token can be well-formed and correctly
       signed but name an account that no longer exists -- a reset database, or
       an expired token. Both return an identical 401 by design. Without this
       the UI renders the error forever and the only escape is clearing
       localStorage by hand. */
    setToken("stale.but.wellformed");
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        mockResponse(401, { detail: "Could not validate credentials" }),
      ),
    );

    await expect(api.nse()).rejects.toThrow();
    await expect(
      api.currency({
        tickers: ["SPY"],
        start_date: "2020-01-01",
        end_date: "2024-01-01",
      }),
    ).rejects.toThrow(ApiError);

    expect(getToken()).toBeNull();
  });

  it("notifies subscribers so the UI can drop back to sign-in", async () => {
    setToken("stale");
    const seen: (string | null)[] = [];
    const { subscribeToToken } = await import("./auth");
    subscribeToToken((token) => seen.push(token));

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => mockResponse(401, { detail: "nope" })),
    );

    await expect(
      api.backtest({
        tickers: ["SPY", "AAPL"],
        start_date: "2020-01-01",
        end_date: "2024-01-01",
        algorithm: "hrp",
        max_weight: 0.5,
        training_days: 252,
        rebalance_every: 21,
        transaction_cost_bps: 10,
      }),
    ).rejects.toThrow();

    expect(seen).toContain(null);
  });

  it("does NOT sign the user out when login itself returns 401", async () => {
    /* A wrong password answers 401 on an unauthenticated endpoint. Clearing
       the token mid-login attempt would be nonsense. */
    setToken("a-valid-session");
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        mockResponse(401, { detail: "Invalid email or password" }),
      ),
    );

    await expect(api.login("a@b.com", "wrong")).rejects.toThrow(ApiError);

    expect(getToken()).toBe("a-valid-session");
  });

  it("leaves the token alone on other error statuses", async () => {
    setToken("still-good");
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => mockResponse(422, { detail: "bad tickers" })),
    );

    await expect(
      api.currency({
        tickers: ["SPY"],
        start_date: "2020-01-01",
        end_date: "2024-01-01",
      }),
    ).rejects.toThrow(ApiError);

    expect(getToken()).toBe("still-good");
  });

  it("leaves the token alone on success", async () => {
    setToken("good");
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => mockResponse(200, { tickers_known: 71 })),
    );

    await api.nse();

    expect(getToken()).toBe("good");
  });
});

describe("error reporting", () => {
  beforeEach(() => {
    localStorage.clear();
    resetTokenListeners();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("surfaces the API's detail string", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => mockResponse(422, { detail: "max_weight is infeasible" })),
    );

    await expect(api.nse()).rejects.toThrow("max_weight is infeasible");
  });

  it("flattens FastAPI field-validation errors into one message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        mockResponse(422, {
          detail: [
            { loc: ["body", "tickers"], msg: "Need at least 2 distinct tickers" },
            { loc: ["body", "end_date"], msg: "must be after start_date" },
          ],
        }),
      ),
    );

    await expect(api.nse()).rejects.toThrow(
      /tickers: Need at least 2 distinct tickers; end_date: must be after start_date/,
    );
  });

  it("explains an unreachable API rather than leaking a fetch error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );

    await expect(api.nse()).rejects.toThrow(/Cannot reach the API/);
  });

  it("reports a status even when the body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error("not json");
        },
      })),
    );

    await expect(api.nse()).rejects.toThrow(/500/);
  });
});
