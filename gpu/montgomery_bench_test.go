//go:build cgo && gpu

package gpu

import (
	"testing"

	"github.com/luxfi/lattice/v7/ring"
)

// BenchmarkMontGPU_Forward measures the Montgomery-form Metal NTT in
// single-poly mode. Compare against BenchmarkRingNTT_Forward (pure Go
// in the ring package) for the per-N crossover analysis.
func BenchmarkMontGPU_Forward(b *testing.B) {
	for _, N := range []int{1024, 2048, 4096, 8192, 16384} {
		b.Run("", func(b *testing.B) {
			r, err := ring.NewRing(N, []uint64{mtQ})
			if err != nil {
				b.Fatalf("NewRing: %v", err)
			}
			s := r.SubRings[0]
			ctx, err := NewMontgomeryNTTContext(s)
			if err != nil {
				b.Fatalf("NewMontgomeryNTTContext: %v", err)
			}
			defer ctx.Close()

			data := genVec(0xCAFE, N, mtQ)
			work := make([]uint64, N)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(work, data)
				if err := ctx.Forward(work, 1); err != nil {
					b.Fatalf("Forward: %v", err)
				}
			}
		})
	}
}

// BenchmarkMontGPU_BatchForward measures the per-poly amortised cost
// when batching B polynomials in a single GPU dispatch. This is the
// "14× speedup" claim in #88.
func BenchmarkMontGPU_BatchForward(b *testing.B) {
	configs := []struct {
		N int
		B int
	}{
		{1024, 16}, {1024, 64},
		{2048, 16}, {2048, 64},
		{4096, 16}, {4096, 64}, {4096, 128},
		{8192, 16}, {8192, 64}, {8192, 128},
	}
	for _, cfg := range configs {
		b.Run("", func(b *testing.B) {
			r, err := ring.NewRing(cfg.N, []uint64{mtQ})
			if err != nil {
				b.Fatalf("NewRing: %v", err)
			}
			s := r.SubRings[0]
			ctx, err := NewMontgomeryNTTContext(s)
			if err != nil {
				b.Fatalf("NewMontgomeryNTTContext: %v", err)
			}
			defer ctx.Close()

			seed := genVec(0xCAFE, cfg.N, mtQ)
			work := make([]uint64, cfg.N*cfg.B)
			for j := 0; j < cfg.B; j++ {
				copy(work[j*cfg.N:(j+1)*cfg.N], seed)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := ctx.Forward(work, uint32(cfg.B)); err != nil {
					b.Fatalf("Forward batched: %v", err)
				}
			}
			b.ReportMetric(float64(b.N)*float64(cfg.B)/b.Elapsed().Seconds(), "ntts/sec")
		})
	}
}

// BenchmarkRingNTT_Forward is the pure-Go baseline.
func BenchmarkRingNTT_Forward(b *testing.B) {
	for _, N := range []int{1024, 2048, 4096, 8192, 16384} {
		b.Run("", func(b *testing.B) {
			r, err := ring.NewRing(N, []uint64{mtQ})
			if err != nil {
				b.Fatalf("NewRing: %v", err)
			}
			s := r.SubRings[0]
			data := genVec(0xCAFE, N, mtQ)
			work := make([]uint64, N)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(work, data)
				s.NTT(work, work)
			}
		})
	}
}
