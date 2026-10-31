import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = [
  ...nextVitals,
  ...nextTs,
  {
    // CommonJS script run by Node to resolve the machine's Playwright install; not part of the app bundle
    files: ["e2e/**/*.cjs"],
    rules: { "@typescript-eslint/no-require-imports": "off" },
  },
  {
    ignores: ["node_modules/**", ".next/**", "out/**", "build/**", "coverage/**", "next-env.d.ts", "src/lib/api/schema.d.ts"],
  },
];

export default eslintConfig;
