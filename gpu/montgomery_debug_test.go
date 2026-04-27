//go:build cgo && gpu

package gpu

import (
	"testing"

	"github.com/luxfi/lattice/v7/ring"
)

// TestDebug isolates the N=4096 divergence with minimal input.
func TestDebug_N4096_OneHot(t *testing.T) {
	r, err := ring.NewRing(4096, []uint64{mtQ})
	if err != nil {
		t.Fatalf("NewRing: %v", err)
	}
	s := r.SubRings[0]

	ctx, err := NewMontgomeryNTTContext(s)
	if err != nil {
		t.Fatalf("NewMontgomeryNTTContext: %v", err)
	}
	defer ctx.Close()

	for _, hotIdx := range []int{0, 1, 2047, 2048, 4095} {
		t.Run("", func(t *testing.T) {
			input := make([]uint64, 4096)
			input[hotIdx] = 1
			goOut := make([]uint64, 4096)
			gpuOut := make([]uint64, 4096)
			copy(goOut, input)
			copy(gpuOut, input)
			s.NTT(goOut, goOut)
			if err := ctx.Forward(gpuOut, 1); err != nil {
				t.Fatalf("Forward: %v", err)
			}
			diff := 0
			firstDiff := -1
			for i := 0; i < 4096; i++ {
				if goOut[i] != gpuOut[i] {
					if firstDiff < 0 {
						firstDiff = i
					}
					diff++
				}
			}
			if diff > 0 {
				t.Errorf("hot=%d differ at %d coefficients (first at i=%d): go=%016x gpu=%016x",
					hotIdx, diff, firstDiff, goOut[firstDiff], gpuOut[firstDiff])
			} else {
				t.Logf("hot=%d byte-equal", hotIdx)
			}
		})
	}
}

// Also test N=2048 to confirm sanity.
func TestDebug_N2048_OneHot(t *testing.T) {
	r, err := ring.NewRing(2048, []uint64{mtQ})
	if err != nil {
		t.Fatalf("NewRing: %v", err)
	}
	s := r.SubRings[0]
	ctx, err := NewMontgomeryNTTContext(s)
	if err != nil {
		t.Fatalf("NewMontgomeryNTTContext: %v", err)
	}
	defer ctx.Close()

	input := make([]uint64, 2048)
	input[0] = 1
	goOut := make([]uint64, 2048)
	gpuOut := make([]uint64, 2048)
	copy(goOut, input)
	copy(gpuOut, input)
	s.NTT(goOut, goOut)
	if err := ctx.Forward(gpuOut, 1); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for i := 0; i < 2048; i++ {
		if goOut[i] != gpuOut[i] {
			t.Errorf("differ at i=%d: go=%016x gpu=%016x", i, goOut[i], gpuOut[i])
			return
		}
	}
}
