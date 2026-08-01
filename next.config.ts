import type { NextConfig } from "next";

const isGitHubPages = process.env.GITHUB_PAGES === "true";

const nextConfig: NextConfig = {
  ...(isGitHubPages
    ? {
        output: "export",
        basePath: "/MIND",
        assetPrefix: "/MIND/",
        trailingSlash: true,
        images: { unoptimized: true },
        // The static project page does not import the Cloudflare-only D1 code.
        // Its `cloudflare:workers` types are available only in the Sites build.
        typescript: { ignoreBuildErrors: true },
      }
    : {}),
};

export default nextConfig;
