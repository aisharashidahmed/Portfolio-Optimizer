import type { Metadata } from "next";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Portfolio Optimizer — Nairobi & US markets",
  description:
    "Portfolio construction and backtesting for investors holding assets across the Nairobi Securities Exchange and US equities, with explicit currency-risk decomposition.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="shell">{children}</main>
      </body>
    </html>
  );
}
