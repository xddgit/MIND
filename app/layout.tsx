import type { Metadata } from "next";
import "./globals.css";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "https://mind-manifold-research.dv6zp9tj87.chatgpt.site";

export const metadata: Metadata = {
    title: "MIND · Diffusion on the Data Manifold",
    description:
      "MIND explicitly models data manifold geometry for high-fidelity diffusion image generation.",
    icons: {
      icon: `${basePath}/favicon.svg`,
      shortcut: `${basePath}/favicon.svg`,
    },
    openGraph: {
      title: "MIND · Diffusion on the Data Manifold",
      description: "Image generation with explicit modeling of data manifold geometry.",
      type: "website",
      images: [{ url: `${siteUrl}${basePath}/og.png`, width: 1200, height: 630, alt: "MIND — Diffusion on the Data Manifold" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "MIND · Diffusion on the Data Manifold",
      description: "Image generation with explicit modeling of data manifold geometry.",
      images: [`${siteUrl}${basePath}/og.png`],
    },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
