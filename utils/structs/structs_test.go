package structs

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"golang.org/x/exp/constraints"

	"github.com/luxfi/lattice/v7/utils/buffer"
)

func TestStructs(t *testing.T) {
	t.Run("Vector/W64/Serialization&Equatable", func(t *testing.T) {
		testVector[uint64](t)
	})

	t.Run("Vector/W32/Serialization&Equatable", func(t *testing.T) {
		testVector[uint32](t)
	})

	t.Run("Vector/W16/Serialization&Equatable", func(t *testing.T) {
		testVector[uint16](t)
	})

	t.Run("Vector/W8/Serialization&Equatable", func(t *testing.T) {
		testVector[uint8](t)
	})

	t.Run("Matrix/W64/Serialization&Equatable", func(t *testing.T) {
		testMatrix[float64](t)
	})

	t.Run("Matrix/W32/Serialization&Equatable", func(t *testing.T) {
		testMatrix[float64](t)
	})

	t.Run("Matrix/W16/Serialization&Equatable", func(t *testing.T) {
		testMatrix[float64](t)
	})

	t.Run("Matrix/W8/Serialization&Equatable", func(t *testing.T) {
		testMatrix[float64](t)
	})
}

func testVector[T constraints.Float | constraints.Integer](t *testing.T) {
	v := Vector[T](make([]T, 64))
	for i := range v {
		v[i] = T(i)
	}
	data, err := v.MarshalBinary()
	require.NoError(t, err)
	vNew := Vector[T]{}
	require.NoError(t, vNew.UnmarshalBinary(data))
	require.True(t, cmp.Equal(v, vNew)) // also tests Equatable
}

func testMatrix[T constraints.Float | constraints.Integer](t *testing.T) {
	v := Matrix[T](make([][]T, 64))
	for i := range v {
		vi := make([]T, 64)
		for j := range vi {
			vi[j] = T(i & j)
		}

		v[i] = vi
	}

	data, err := v.MarshalBinary()
	require.NoError(t, err)
	vNew := Matrix[T]{}
	require.NoError(t, vNew.UnmarshalBinary(data))
	require.True(t, cmp.Equal(v, vNew)) // also tests Equatable
}

// TestVectorReadFrom_RejectsOversizedLengthPrefix is a regression test for
// luxfi/lattice#4 (LUX-517): Vector[T].ReadFrom previously did `make([]T, size)`
// against an attacker-controlled 8-byte length prefix without any bound check,
// causing an unrecoverable runtime OOM ("cannot allocate N-byte block") on a
// 9-byte crafted input. The bound check now rejects oversized length prefixes
// with buffer.ErrSliceTooLarge before any allocation.
//
// Discovered by FuzzPulsarSign1Round1Data in github.com/luxfi/pulsar,
// failing seed testdata/fuzz/FuzzPulsarSign1Round1Data/a80b8d313b40fa55.
func TestVectorReadFrom_RejectsOversizedLengthPrefix(t *testing.T) {
	// 8-byte LE length = 0x40005AD893AD = 70_368_955_777_453 entries.
	// Pre-fix this triggered `make([]uint64, 70T)` → unrecoverable OOM.
	raw := []byte("\xad\x93\xd8\x5a\x00\x04\x00\x00\x5c")
	var v Vector[uint64]
	err := v.UnmarshalBinary(raw)
	require.Error(t, err)
	require.ErrorIs(t, err, buffer.ErrSliceTooLarge)
}

// TestMatrixReadFrom_RejectsOversizedRowsPrefix mirrors the Vector check for
// Matrix[T].ReadFrom: an attacker-controlled rows count would otherwise hit
// `make([][]T, rows)` without bound. Same fix, same error semantics.
func TestMatrixReadFrom_RejectsOversizedRowsPrefix(t *testing.T) {
	raw := []byte("\xad\x93\xd8\x5a\x00\x04\x00\x00\x5c")
	var m Matrix[uint64]
	err := m.UnmarshalBinary(raw)
	require.Error(t, err)
	require.ErrorIs(t, err, buffer.ErrSliceTooLarge)
}

// TestVectorReadFrom_RespectsConfiguredCap verifies that the bound check
// honors buffer.SetMaxSliceLen so callers (e.g. luxfi/warp) can tighten the
// limit without modifying lattice. A length-prefix one above the configured
// cap must be rejected; a length-prefix exactly at the cap remains accepted
// (subject to the underlying readers having enough bytes).
func TestVectorReadFrom_RespectsConfiguredCap(t *testing.T) {
	orig := buffer.MaxSliceLen()
	t.Cleanup(func() { buffer.SetMaxSliceLen(orig) })

	buffer.SetMaxSliceLen(8)

	// 9 entries — one above the cap — should reject.
	prefix := []byte{0x09, 0, 0, 0, 0, 0, 0, 0}
	var v Vector[uint64]
	err := v.UnmarshalBinary(prefix)
	require.ErrorIs(t, err, buffer.ErrSliceTooLarge)
}
