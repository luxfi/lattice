import Link from "next/link"

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center px-4">
      <div className="container flex flex-col items-center gap-12 py-24 sm:gap-16 sm:py-32">
        <div className="flex flex-col items-center gap-4 text-center">
          <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
            Documentation
          </h1>
          <p className="max-w-2xl text-lg text-muted-foreground">
            Get started with our comprehensive documentation
          </p>
        </div>
        <div className="flex gap-4">
          <Link
            href="/docs"
            className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
          >
            Get Started
          </Link>
        </div>
      </div>
    </main>
  )
}
