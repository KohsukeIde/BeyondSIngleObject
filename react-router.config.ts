import type { Config } from "@react-router/dev/config";

export default {
  // Config options...
  appDirectory: "src/app",
  // Server-side render by default, to enable SPA mode set this to `false`
  ssr: false,
  // Must match Vite's `base`. In production the site is served from the
  // GitHub Pages subpath; without this the client router fails to match
  // "/BeyondSingleObject/" and renders the 404 ErrorBoundary.
  basename:
    process.env.NODE_ENV === "production" ? "/BeyondSingleObject/" : "/",
  // async prerender() {
  //   return ["/popular"];
  // },
} satisfies Config;