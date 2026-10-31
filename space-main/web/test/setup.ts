import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";

import { resetCsrfForTests } from "@/lib/api/client";

// jsdom has no ResizeObserver; Recharts' ResponsiveContainer only needs the constructor to exist.
class NoopResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= NoopResizeObserver as unknown as typeof ResizeObserver;

// No network in frontend tests: any fetch that a test did not replace with the fake API fails loudly.
beforeEach(() => {
  vi.stubGlobal("fetch", () => {
    throw new Error("network access attempted in a frontend test");
  });
  window.sessionStorage.clear();
  resetCsrfForTests();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
