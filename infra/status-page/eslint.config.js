// ESLint flat config for the marin-infra-dashboard package.
//
// Two environments share one config:
// - server/** runs under Node, imports from @hono, @google-cloud/compute
// - web/**    runs in the browser, uses React + hooks
//
// The base (recommended) TS rules apply to both. The React-hooks and
// react-refresh plugins are scoped to web/**/*.{ts,tsx} since the server
// never touches React.
//
// Type-checked linting is off by default to keep CI fast. Upgrade to
// tseslint.configs.recommendedTypeChecked if we start caring about rules
// that need type information (e.g. no-floating-promises, no-misused-promises).

import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: ["server/dist", "web/dist", "node_modules"],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["server/**/*.ts"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        ...globals.node,
      },
    },
  },
  {
    // PostCSS / build-tool configs that still live on CommonJS.
    files: ["**/*.cjs"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "commonjs",
      globals: {
        ...globals.node,
      },
    },
  },
  {
    files: ["web/**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        ...globals.browser,
      },
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
    },
  },
  {
    // Project-wide overrides — keep the defaults sensible for a small
    // internal dashboard. Prefer underscore-prefixed unused args for
    // intentional ignores (matches TS's noUnusedParameters convention).
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
    },
  },
);
