//go:build cgo && gpu

package gpu

import (
	"math/rand/v2"
	"testing"

	"github.com/luxfi/lattice/v7/ring"
)

// Standard test moduli used across the lattice tree. Q60[0] is a 60-bit
// NTT-friendly prime; matches the production parameters in
// luxfi/lattice/v7/core/parameters.
const (
	mtN uint32 = 4096
	mtQ uint64 = 0x1fffffffffe00001 // 2^61 - 2^21 + 1, NTT-friendly for N up to 2^21
)

func newTestSubRing(t *testing.T, N int, Q uint64) *ring.SubRing {
	t.Helper()
	s, err := ring.NewSubRing(N, Q)
	if err != nil {
		t.Fatalf("NewSubRing: %v", err)
	}
	// generateNTTConstants is unexported; the public path is via
	// ring.NewRing -> SubRing constructor which calls it. Use the
	// SetGPUDispatchers entry point side-effect: any SubRing that
	// has not had its NTT constants generated will have NInv == 0.
	// Force generation by calling NTT once in the pure-Go path on a
	// throwaway buffer of the right shape.
	_ = s
	r, err := ring.NewRing(N, []uint64{Q})
	if err != nil {
		t.Fatalf("NewRing: %v", err)
	}
	return r.SubRings[0]
}

// genVec returns a length-N slice with bytes drawn from a deterministic
// PRNG. Coefficients are reduced mod Q so they live in the canonical
// SubRing.NTT input domain.
func genVec(seed uint64, N int, Q uint64) []uint64 {
	rng := rand.New(rand.NewPCG(seed, seed^0xdeadbeef))
	v := make([]uint64, N)
	for i := range v {
		v[i] = rng.Uint64() % Q
	}
	return v
}

// TestMontgomery_ByteEqual_Forward verifies that the GPU/CPU Montgomery
// fallback produces byte-equal output to ring.SubRing.NTT (pure Go) for
// 1024 distinct random vectors.
func TestMontgomery_ByteEqual_Forward(t *testing.T) {
	const vectors = 1024
	for _, N := range []int{1024, 2048, 4096} {
		t.Run("", func(t *testing.T) {
			s := newTestSubRing(t, N, mtQ)
			ctx, err := NewMontgomeryNTTContext(s)
			if err != nil {
				t.Fatalf("NewMontgomeryNTTContext: %v", err)
			}
			defer ctx.Close()

			for v := 0; v < vectors; v++ {
				input := genVec(uint64(v)*131+uint64(N), N, mtQ)
				goOutput := make([]uint64, N)
				gpuOutput := make([]uint64, N)
				copy(goOutput, input)
				copy(gpuOutput, input)

				s.NTT(goOutput, goOutput)
				if err := ctx.Forward(gpuOutput, 1); err != nil {
					t.Fatalf("ctx.Forward: %v", err)
				}

				for i := 0; i < N; i++ {
					if goOutput[i] != gpuOutput[i] {
						t.Fatalf("vec=%d N=%d byte differ at i=%d: go=%016x gpu=%016x",
							v, N, i, goOutput[i], gpuOutput[i])
					}
				}
			}
		})
	}
}

// TestMontgomery_ByteEqual_Inverse verifies INTT byte-equality.
func TestMontgomery_ByteEqual_Inverse(t *testing.T) {
	const vectors = 1024
	for _, N := range []int{1024, 2048, 4096} {
		t.Run("", func(t *testing.T) {
			s := newTestSubRing(t, N, mtQ)
			ctx, err := NewMontgomeryNTTContext(s)
			if err != nil {
				t.Fatalf("NewMontgomeryNTTContext: %v", err)
			}
			defer ctx.Close()

			for v := 0; v < vectors; v++ {
				input := genVec(uint64(v)*131+uint64(N)+0x99, N, mtQ)

				// Pre-NTT both copies so we are inverting NTT-domain values.
				goNTT := make([]uint64, N)
				gpuNTT := make([]uint64, N)
				copy(goNTT, input)
				copy(gpuNTT, input)
				s.NTT(goNTT, goNTT)
				s.NTT(gpuNTT, gpuNTT)

				goOut := make([]uint64, N)
				gpuOut := make([]uint64, N)
				copy(goOut, goNTT)
				copy(gpuOut, gpuNTT)

				s.INTT(goOut, goOut)
				if err := ctx.Backward(gpuOut, 1); err != nil {
					t.Fatalf("ctx.Backward: %v", err)
				}

				for i := 0; i < N; i++ {
					if goOut[i] != gpuOut[i] {
						t.Fatalf("vec=%d N=%d byte differ at i=%d: go=%016x gpu=%016x",
							v, N, i, goOut[i], gpuOut[i])
					}
				}
			}
		})
	}
}

// TestMontgomery_RoundTrip verifies INTT(NTT(x)) == x for the GPU path.
func TestMontgomery_RoundTrip(t *testing.T) {
	for _, N := range []int{1024, 2048, 4096, 8192} {
		t.Run("", func(t *testing.T) {
			s := newTestSubRing(t, N, mtQ)
			ctx, err := NewMontgomeryNTTContext(s)
			if err != nil {
				t.Fatalf("NewMontgomeryNTTContext: %v", err)
			}
			defer ctx.Close()

			input := genVec(0xC0DEFACE, N, mtQ)
			work := make([]uint64, N)
			copy(work, input)

			if err := ctx.Forward(work, 1); err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if err := ctx.Backward(work, 1); err != nil {
				t.Fatalf("Backward: %v", err)
			}

			for i := 0; i < N; i++ {
				if work[i] != input[i] {
					t.Fatalf("N=%d round-trip mismatch at i=%d: got %016x want %016x",
						N, i, work[i], input[i])
				}
			}
		})
	}
}

// TestBatchNTT_NoSegfault verifies that the BatchNTT contract that
// previously SIGSEGV'd at every (N>=1024, B>=16) configuration now
// completes without crashing AND produces output byte-equal to running
// per-poly Montgomery NTTs serially.
func TestBatchNTT_NoSegfault(t *testing.T) {
	configs := []struct {
		N int
		B int
	}{
		{1024, 16},
		{2048, 16},
		{4096, 16},
		{4096, 32},
		{4096, 128},
		{8192, 16},
	}
	for _, cfg := range configs {
		t.Run("", func(t *testing.T) {
			s := newTestSubRing(t, cfg.N, mtQ)
			ctx, err := NewMontgomeryNTTContext(s)
			if err != nil {
				t.Fatalf("NewMontgomeryNTTContext: %v", err)
			}
			defer ctx.Close()

			// Build per-batch inputs.
			polys := make([][]uint64, cfg.B)
			batchBuf := make([]uint64, cfg.N*cfg.B)
			for b := 0; b < cfg.B; b++ {
				polys[b] = genVec(uint64(b)*0x1234567+uint64(cfg.N), cfg.N, mtQ)
				copy(batchBuf[b*cfg.N:(b+1)*cfg.N], polys[b])
			}

			// Batched dispatch (the segfault path).
			if err := ctx.Forward(batchBuf, uint32(cfg.B)); err != nil {
				t.Fatalf("Forward batched: %v", err)
			}

			// Compare each batch slice to the per-poly Go output.
			for b := 0; b < cfg.B; b++ {
				goOut := make([]uint64, cfg.N)
				copy(goOut, polys[b])
				s.NTT(goOut, goOut)
				gpuSlice := batchBuf[b*cfg.N : (b+1)*cfg.N]
				for i := 0; i < cfg.N; i++ {
					if goOut[i] != gpuSlice[i] {
						t.Fatalf("N=%d B=%d b=%d byte differ at i=%d: go=%016x gpu=%016x",
							cfg.N, cfg.B, b, i, goOut[i], gpuSlice[i])
					}
				}
			}
		})
	}
}
