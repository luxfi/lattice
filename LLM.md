# AI Assistant Knowledge Base

**Last Updated**: 2026-03-22
**Project**: lattice
**Organization**: luxfi

## Project Overview

Post-quantum lattice-based homomorphic encryption library. Go module `github.com/luxfi/lattice/v7` (Go 1.26.1).

## Essential Commands

### Development
```bash
# Build
go build ./...

# Test
go test -count=1 ./...

# Test with SIMD (amd64 only, requires AVX2)
GOEXPERIMENT=simd go test -count=1 ./ring/ -v -run 'TestNTT|TestMRedLazy4|TestMul64x4|TestNTTSimd'

# Static checks
make checks

# Benchmark NTT
go test -bench=BenchmarkNTT -benchmem ./ring/
```

## Architecture

### NTT (Number Theoretic Transform) - ring/ntt.go
- Core butterfly operations use Montgomery reduction (`MRedLazy`) with 128-bit multiply
- Loop-unrolled by 8/16 using `unsafe.Pointer` casts to fixed-size arrays
- Two NTT variants: Standard (nega-cyclic) and ConjugateInvariant (Z[X+X^-1])
- Each has Forward/Backward (exact) and ForwardLazy/BackwardLazy (approximate bounds)
- Dispatch: `nttCoreLazy` -> `nttUnrolled16Lazy` (N>=16) or `nttLazy` (N<16)

### SIMD Acceleration - ring/ntt_simd.go (GOEXPERIMENT=simd, amd64)
- Build tags: `goexperiment.simd && amd64` (SIMD path) / `!(goexperiment.simd && amd64)` (stub)
- Uses Go 1.26 `simd/archsimd` package with `Uint64x4` (AVX2 256-bit vectors)
- Montgomery reduction vectorized via `Uint32x8.MulEvenWiden` (VPMULUDQ) decomposition
  - 64x64->128 multiply emulated with 4x VPMULUDQ + shifts + adds
  - `mredLazy4`: 4 parallel MRedLazy operations
  - `mul64x4`: 4 parallel 64x64->128 multiplies
- SIMD path accelerates t>=8 butterfly stages (dominate runtime for large N)
- Small-t stages (t=4, t=2, t=1) remain scalar (strided access patterns)
- Runtime AVX2 check via `archsimd.X86.AVX2()`
- Transparent fallback: `nttCoreLazyAccel` returns false -> scalar path runs

### Key archsimd types for NTT work
- `Uint64x4`: Add/Sub (AVX2), Mul (AVX-512 only, VPMULLQ)
- `Uint32x8.MulEvenWiden` -> `Uint64x4` (AVX2, VPMULUDQ) - key primitive
- No `MulHi` for uint64 - must decompose via 32-bit widening
- `GreaterEqual` + `Masked` for conditional reduction (replaces scalar branch)

## Key Technologies

- Go 1.26 with GOEXPERIMENT=simd for SIMD intrinsics
- Montgomery form arithmetic for modular operations (ring/modular_reduction.go)
- Barrett reduction for modular reduction (BRed/BRedAdd)
- NTT primes: 60-bit primes (Q ~ 2^60) in Montgomery form

## Development Workflow

- Tests use known test vectors (ring/ntt_test.go) for N=16,32,64,128,256,512
- CI: Go 1.25 + 1.26 standard tests, plus GOEXPERIMENT=simd job on ubuntu (amd64)
- Static checks: gofmt, govet, goimports, staticcheck, govulncheck, gosec

## Context for All AI Assistants

This file (`LLM.md`) is symlinked as:
- `.AGENTS.md`
- `CLAUDE.md`
- `QWEN.md`
- `GEMINI.md`

All files reference the same knowledge base. Updates here propagate to all AI systems.

## Rules for AI Assistants

1. **ALWAYS** update LLM.md with significant discoveries
2. **NEVER** commit symlinked files (.AGENTS.md, CLAUDE.md, etc.) - they're in .gitignore
3. **NEVER** create random summary files - update THIS file

---

**Note**: This file serves as the single source of truth for all AI assistants working on this project.
