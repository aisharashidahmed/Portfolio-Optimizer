"use client";

import { useEffect, useState } from "react";
import { ApiError, api, getToken, setToken } from "@/lib/api";

/**
 * Minimal auth wrapper.
 *
 * Every analysis endpoint requires a token. Rather than hide that, the gate
 * explains it and offers a one-click demo account so a reviewer can see the
 * app without inventing credentials.
 */
export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [authed, setAuthed] = useState(false);
  const [email, setEmail] = useState("demo@example.com");
  const [password, setPassword] = useState("demo-password");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setAuthed(getToken() !== null);
    setReady(true);
  }, []);

  const authenticate = async () => {
    setBusy(true);
    setError(null);
    try {
      // Sign up, falling back to login when the account already exists.
      let token: string;
      try {
        token = (await api.signup(email, password)).access_token;
      } catch (exc) {
        if (exc instanceof ApiError && exc.status === 409) {
          token = (await api.login(email, password)).access_token;
        } else {
          throw exc;
        }
      }
      setToken(token);
      setAuthed(true);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not sign in");
    } finally {
      setBusy(false);
    }
  };

  if (!ready) return null;

  if (!authed) {
    return (
      <div className="card" style={{ maxWidth: 440 }}>
        <h2>Sign in</h2>
        <p className="card-note">
          The analysis endpoints are authenticated. The pre-filled demo account
          works out of the box &mdash; it is created on first use.
        </p>

        <div className="stack-sm" style={{ marginTop: 16 }}>
          <div className="field">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="username"
            />
          </div>
          <div className="field">
            <label htmlFor="password">Password (8+ characters)</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
            />
          </div>
        </div>

        {error && (
          <div className="error" style={{ marginTop: 14 }}>
            {error}
          </div>
        )}

        <button
          type="button"
          className="btn"
          onClick={authenticate}
          disabled={busy}
          style={{ marginTop: 14 }}
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </div>
    );
  }

  return <>{children}</>;
}
