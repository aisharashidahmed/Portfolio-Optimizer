/**
 * Test environment setup.
 *
 * happy-dom 19 exposes the `Storage` constructor but does not install a
 * `localStorage` instance on the global, so the bare `localStorage` reference
 * the app uses is undefined under test. Instantiating happy-dom's own class
 * gives a real Storage whose methods live on `Storage.prototype`, which is
 * what the auth tests spy on to simulate a private window.
 */
if (typeof globalThis.localStorage === "undefined") {
  Object.defineProperty(globalThis, "localStorage", {
    value: new Storage(),
    writable: true,
    configurable: true,
  });
}
