/**
 * Auth token store.
 *
 * Owns the token and notifies subscribers when it changes, so a 401 anywhere
 * in the app can drop every mounted component back to the signed-in state
 * without prop-drilling or a context provider.
 *
 * Why this exists: a stored token can stop working while remaining perfectly
 * well-formed -- it expires, or the account it names is deleted (a reset
 * database, say). Both produce an identical 401, deliberately, so the endpoint
 * cannot be used to probe which accounts exist. Without this the UI renders
 * that error forever and the only escape is clearing localStorage by hand.
 */

const TOKEN_KEY = "portfolio.token";

type Listener = (token: string | null) => void;

const listeners = new Set<Listener>();

/** localStorage throws in a private window; never let that break a render. */
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
    /* ignore: the in-memory notification below still fires */
  }
  for (const listener of listeners) listener(token);
}

export function signOut(): void {
  setToken(null);
}

/** Subscribe to token changes. Returns an unsubscribe function. */
export function subscribeToToken(listener: Listener): () => void {
  listeners.add(listener);

  // A second tab signing in or out writes to the same key; `storage` fires in
  // the *other* tabs, so this keeps them consistent too.
  const onStorage = (event: StorageEvent) => {
    if (event.key === TOKEN_KEY || event.key === null) listener(getToken());
  };
  window.addEventListener("storage", onStorage);

  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** Test seam: drop every subscriber. */
export function resetTokenListeners(): void {
  listeners.clear();
}
