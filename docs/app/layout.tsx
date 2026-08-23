import "./global.css"
import { RootProvider } from "fumadocs-ui/provider/next"
import { ZenSans } from "@hanzo/font/sans"
import { ZenMono } from "@hanzo/font/mono"
import type { ReactNode } from "react"

export const metadata = {
  title: {
    default: "Lattice Crypto Documentation",
    template: "%s | Lattice Crypto",
  },
  description: "Post-quantum lattice-based cryptography",
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${ZenSans.variable} ${ZenMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-svh bg-background font-sans antialiased">
        <RootProvider
          search={{
            enabled: true,
          }}
          theme={{
            enabled: true,
            defaultTheme: "dark",
          }}
        >
          <div className="relative flex min-h-svh flex-col bg-background">
            {children}
          </div>
        </RootProvider>
      </body>
    </html>
  )
}
