import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const baseUrl = `${protocol}://${host}`;

  return {
    title: "MIND · Diffusion on the Data Manifold",
    description:
      "MIND explicitly models data manifold geometry for high-fidelity diffusion image generation.",
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
    openGraph: {
      title: "MIND · Diffusion on the Data Manifold",
      description: "Image generation with explicit modeling of data manifold geometry.",
      type: "website",
      images: [{ url: `${baseUrl}/og.png`, width: 1200, height: 630, alt: "MIND — Diffusion on the Data Manifold" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "MIND · Diffusion on the Data Manifold",
      description: "Image generation with explicit modeling of data manifold geometry.",
      images: [`${baseUrl}/og.png`],
    },
  };
}

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
