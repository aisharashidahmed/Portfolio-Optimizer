"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getToken, signOut, subscribeToToken } from "@/lib/auth";

const LINKS = [
  { href: "/", label: "Currency risk" },
  { href: "/optimize", label: "Optimize" },
  { href: "/backtest", label: "Backtest" },
  { href: "/nse", label: "NSE data" },
];

export default function Nav() {
  const pathname = usePathname();
  const [signedIn, setSignedIn] = useState(false);

  useEffect(() => {
    setSignedIn(getToken() !== null);
    return subscribeToToken((token) => setSignedIn(token !== null));
  }, []);

  return (
    <nav className="nav">
      <div className="nav-inner">
        <Link href="/" className="nav-brand">
          Portfolio Optimizer
        </Link>
        <div className="nav-links">
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="nav-link"
              aria-current={pathname === link.href ? "page" : undefined}
            >
              {link.label}
            </Link>
          ))}
        </div>

        {signedIn && (
          <>
            <span className="nav-spacer" />
            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: "0.8rem", padding: "5px 12px" }}
              onClick={signOut}
            >
              Sign out
            </button>
          </>
        )}
      </div>
    </nav>
  );
}
