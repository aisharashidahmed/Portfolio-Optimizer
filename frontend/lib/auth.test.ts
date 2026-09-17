import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getToken,
  resetTokenListeners,
  setToken,
  signOut,
  subscribeToToken,
} from "./auth";

describe("token storage", () => {
  beforeEach(() => {
    localStorage.clear();
    resetTokenListeners();
  });

  it("round-trips a token", () => {
    setToken("abc.def.ghi");
    expect(getToken()).toBe("abc.def.ghi");
  });

  it("returns null when nothing is stored", () => {
    expect(getToken()).toBeNull();
  });

  it("clears the token on sign out", () => {
    setToken("abc");
    signOut();
    expect(getToken()).toBeNull();
  });

  it("survives localStorage throwing", () => {
    // A private window, or blocked site data. The app must still render.
    // Spy on the instance, not Storage.prototype: happy-dom's localStorage
    // does not route its methods through the prototype, so a prototype spy
    // never intercepts.
    const getItem = vi
      .spyOn(localStorage, "getItem")
      .mockImplementation(() => {
        throw new Error("blocked");
      });
    const setItem = vi
      .spyOn(localStorage, "setItem")
      .mockImplementation(() => {
        throw new Error("blocked");
      });

    expect(() => setToken("abc")).not.toThrow();
    expect(getToken()).toBeNull();

    getItem.mockRestore();
    setItem.mockRestore();
  });
});

describe("subscriptions", () => {
  beforeEach(() => {
    localStorage.clear();
    resetTokenListeners();
  });

  it("notifies subscribers when a token is set", () => {
    const seen: (string | null)[] = [];
    subscribeToToken((token) => seen.push(token));

    setToken("abc");
    expect(seen).toEqual(["abc"]);
  });

  it("notifies subscribers when the token is cleared", () => {
    const seen: (string | null)[] = [];
    subscribeToToken((token) => seen.push(token));

    setToken("abc");
    signOut();
    expect(seen).toEqual(["abc", null]);
  });

  it("notifies every subscriber", () => {
    const first = vi.fn();
    const second = vi.fn();
    subscribeToToken(first);
    subscribeToToken(second);

    setToken("abc");

    expect(first).toHaveBeenCalledWith("abc");
    expect(second).toHaveBeenCalledWith("abc");
  });

  it("stops notifying after unsubscribe", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeToToken(listener);

    unsubscribe();
    setToken("abc");

    expect(listener).not.toHaveBeenCalled();
  });

  it("still notifies when localStorage throws", () => {
    /* The in-memory notification is what drives the UI, so it must not
       depend on storage succeeding. */
    const setItem = vi
      .spyOn(localStorage, "setItem")
      .mockImplementation(() => {
        throw new Error("blocked");
      });
    const listener = vi.fn();
    subscribeToToken(listener);

    setToken("abc");

    expect(listener).toHaveBeenCalledWith("abc");
    setItem.mockRestore();
  });

  it("reacts to another tab changing the token", () => {
    const listener = vi.fn();
    subscribeToToken(listener);

    localStorage.setItem("portfolio.token", "from-other-tab");
    window.dispatchEvent(
      new StorageEvent("storage", { key: "portfolio.token" }),
    );

    expect(listener).toHaveBeenCalledWith("from-other-tab");
  });

  it("ignores unrelated storage keys", () => {
    const listener = vi.fn();
    subscribeToToken(listener);

    window.dispatchEvent(new StorageEvent("storage", { key: "theme" }));

    expect(listener).not.toHaveBeenCalled();
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});
