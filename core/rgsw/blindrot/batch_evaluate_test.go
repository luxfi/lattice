package blindrot

import (
	"bytes"
	"testing"

	"github.com/luxfi/lattice/v7/core/rlwe"
	"github.com/luxfi/lattice/v7/ring"
	"github.com/luxfi/lattice/v7/utils"
	"github.com/stretchr/testify/require"
)

// batchHarness builds the parameter set and shared inputs used by every
// batch test. Mirrors testBlindRotation in blindrot_test.go.
type batchHarness struct {
	paramsBR  rlwe.Parameters
	paramsLWE rlwe.Parameters
	eval      *Evaluator
	BRK       MemBlindRotationEvaluationKeySet
	ctLWE     *rlwe.Ciphertext
	testPoly  ring.Poly
}

func newBatchHarness(t testing.TB) *batchHarness {
	paramsBR, err := rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{
		LogN:    10,
		Q:       []uint64{0x7fff801},
		NTTFlag: true,
	})
	require.NoError(t, err)

	paramsLWE, err := rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{
		LogN:    9,
		Q:       []uint64{0x3001},
		NTTFlag: true,
	})
	require.NoError(t, err)

	evkParams := rlwe.EvaluationKeyParameters{BaseTwoDecomposition: utils.Pointy(7)}

	scaleLWE := float64(paramsLWE.Q()[0]) / 4.0
	scaleBR := float64(paramsBR.Q()[0]) / 4.0
	testPoly := InitTestPolynomial(sign, rlwe.NewScale(scaleBR), paramsBR.RingQ(), -1, 1)

	skLWE := rlwe.NewKeyGenerator(paramsLWE).GenSecretKeyNew()
	encryptorLWE := rlwe.NewEncryptor(paramsLWE, skLWE)

	ptLWE := rlwe.NewPlaintext(paramsLWE, paramsLWE.MaxLevel())
	slots := 16
	values := make([]float64, slots)
	for i := 0; i < slots; i++ {
		values[i] = -1 + float64(2*i)/float64(slots)
	}
	for i := range values {
		if values[i] < 0 {
			ptLWE.Value.Coeffs[0][i] = paramsLWE.Q()[0] - uint64(-values[i]*scaleLWE)
		} else {
			ptLWE.Value.Coeffs[0][i] = uint64(values[i] * scaleLWE)
		}
	}
	if ptLWE.IsNTT {
		paramsLWE.RingQ().NTT(ptLWE.Value, ptLWE.Value)
	}
	ctLWE := rlwe.NewCiphertext(paramsLWE, 1, paramsLWE.MaxLevel())
	require.NoError(t, encryptorLWE.Encrypt(ptLWE, ctLWE))

	skBR := rlwe.NewKeyGenerator(paramsBR).GenSecretKeyNew()
	BRK := GenEvaluationKeyNew(paramsBR, skBR, paramsLWE, skLWE, evkParams)

	return &batchHarness{
		paramsBR:  paramsBR,
		paramsLWE: paramsLWE,
		eval:      NewEvaluator(paramsBR, paramsLWE),
		BRK:       BRK,
		ctLWE:     ctLWE,
		testPoly:  testPoly,
	}
}

// makeBatch returns N copies of the same ciphertext and test-poly map. Each
// Evaluate call against (ctLWE, testPolyMap, BRK) is deterministic for fixed
// inputs, so N copies must produce N byte-equal outputs both serially and in
// batch.
func (h *batchHarness) makeBatch(n int) ([]*rlwe.Ciphertext, []map[int]*ring.Poly) {
	cts := make([]*rlwe.Ciphertext, n)
	polys := make([]map[int]*ring.Poly, n)
	tp := h.testPoly
	for i := 0; i < n; i++ {
		// CopyNew gives each batch slot an independent input so a buggy
		// implementation that aliases through the shared receiver buffers
		// shows up as a byte mismatch.
		cts[i] = h.ctLWE.CopyNew()
		m := make(map[int]*ring.Poly, 16)
		for s := 0; s < 16; s++ {
			m[s] = &tp
		}
		polys[i] = m
	}
	return cts, polys
}

func ctBytesEqual(a, b *rlwe.Ciphertext) bool {
	for i := range a.Value {
		if i >= len(b.Value) {
			return false
		}
		for j := range a.Value[i].Coeffs {
			if !bytes.Equal(uint64SliceBytes(a.Value[i].Coeffs[j]), uint64SliceBytes(b.Value[i].Coeffs[j])) {
				return false
			}
		}
	}
	return a.IsNTT == b.IsNTT
}

func uint64SliceBytes(s []uint64) []byte {
	out := make([]byte, 8*len(s))
	for i, v := range s {
		for k := 0; k < 8; k++ {
			out[8*i+k] = byte(v >> (8 * k))
		}
	}
	return out
}

// TestBatchEvaluate_ByteEqualSerial verifies that BatchEvaluate(N) produces
// the same byte sequence per slot as N serial Evaluate calls on a fresh
// Evaluator.
func TestBatchEvaluate_ByteEqualSerial(t *testing.T) {
	h := newBatchHarness(t)
	const N = 16

	cts, polys := h.makeBatch(N)

	serial := make([]map[int]*rlwe.Ciphertext, N)
	for i := 0; i < N; i++ {
		serialEval := NewEvaluator(h.paramsBR, h.paramsLWE)
		out, err := serialEval.Evaluate(cts[i], polys[i], h.BRK)
		require.NoError(t, err)
		serial[i] = out
	}

	cts2, polys2 := h.makeBatch(N)
	batchEval := NewEvaluator(h.paramsBR, h.paramsLWE)
	parallel, errs := batchEval.BatchEvaluate(cts2, polys2, h.BRK)

	require.Len(t, parallel, N)
	require.Len(t, errs, N)
	for i := 0; i < N; i++ {
		require.NoErrorf(t, errs[i], "batch slot %d", i)
		require.Equalf(t, len(serial[i]), len(parallel[i]), "slot %d map size", i)
		for k, sv := range serial[i] {
			pv, ok := parallel[i][k]
			require.Truef(t, ok, "slot %d missing key %d", i, k)
			require.Truef(t, ctBytesEqual(sv, pv), "slot %d key %d byte mismatch", i, k)
		}
	}
}

func TestBatchEvaluate_Empty(t *testing.T) {
	h := newBatchHarness(t)
	out, errs := h.eval.BatchEvaluate(nil, nil, h.BRK)
	require.Nil(t, out)
	require.Nil(t, errs)
}

func TestBatchEvaluate_Single(t *testing.T) {
	h := newBatchHarness(t)
	cts, polys := h.makeBatch(1)

	cts2, polys2 := h.makeBatch(1)
	serialEval := NewEvaluator(h.paramsBR, h.paramsLWE)
	expected, err := serialEval.Evaluate(cts2[0], polys2[0], h.BRK)
	require.NoError(t, err)

	got, errs := h.eval.BatchEvaluate(cts, polys, h.BRK)
	require.Len(t, got, 1)
	require.Len(t, errs, 1)
	require.NoError(t, errs[0])
	require.Equal(t, len(expected), len(got[0]))
	for k, sv := range expected {
		pv, ok := got[0][k]
		require.Truef(t, ok, "missing key %d", k)
		require.Truef(t, ctBytesEqual(sv, pv), "key %d byte mismatch", k)
	}
}

func TestBatchEvaluate_LengthMismatch(t *testing.T) {
	h := newBatchHarness(t)
	cts, _ := h.makeBatch(4)
	out, errs := h.eval.BatchEvaluate(cts, nil, h.BRK)
	require.Len(t, out, 4)
	require.Len(t, errs, 4)
	for i := range errs {
		require.ErrorIs(t, errs[i], errMismatchedTestPolyLen)
	}
}

// TestBatchEvaluate_RaceClean — runs concurrent BatchEvaluate dispatches.
// Together with `go test -race` this surfaces any unintended sharing of
// receiver buffers across worker goroutines.
func TestBatchEvaluate_RaceClean(t *testing.T) {
	h := newBatchHarness(t)
	const N = 8
	cts, polys := h.makeBatch(N)
	out, errs := h.eval.BatchEvaluate(cts, polys, h.BRK)
	require.Len(t, out, N)
	for i, e := range errs {
		require.NoErrorf(t, e, "slot %d", i)
		require.NotNil(t, out[i])
	}
}

func BenchmarkBatchEvaluate_Serial_N16(b *testing.B) {
	h := newBatchHarness(b)
	cts, polys := h.makeBatch(16)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		eval := NewEvaluator(h.paramsBR, h.paramsLWE)
		for i := 0; i < len(cts); i++ {
			if _, err := eval.Evaluate(cts[i], polys[i], h.BRK); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkBatchEvaluate_Parallel_N16(b *testing.B) {
	h := newBatchHarness(b)
	cts, polys := h.makeBatch(16)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		_, errs := h.eval.BatchEvaluate(cts, polys, h.BRK)
		for _, e := range errs {
			if e != nil {
				b.Fatal(e)
			}
		}
	}
}
