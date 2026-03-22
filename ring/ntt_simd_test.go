//go:build goexperiment.simd && amd64

package ring

import (
	"math/bits"
	"simd/archsimd"
	"testing"
)

// TestMRedLazy4 verifies that the SIMD mredLazy4 produces identical results
// to the scalar MRedLazy for a variety of inputs.
func TestMRedLazy4(t *testing.T) {
	if !simdAVX2 {
		t.Skip("AVX2 not available")
	}

	// Use a realistic NTT prime and its Montgomery constant
	Q := uint64(576460752303439873)
	MRedConst := GenMRedConstant(Q)

	testCases := [][4]uint64{
		{0, 0, 0, 0},
		{1, 1, 1, 1},
		{Q - 1, Q - 1, Q - 1, Q - 1},
		{Q / 2, Q / 3, Q / 4, Q / 5},
		{123456789, 987654321, 1<<32 - 1, 1<<62 - 1},
		{0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF, 0x1, 0xDEADBEEFCAFEBABE},
	}

	yArr := [4]uint64{
		42, 1<<40 + 7, Q - 3, 0xCAFEBABE,
	}

	for _, xArr := range testCases {
		result := mredLazy4(&xArr, &yArr, Q, MRedConst)

		var out [4]uint64
		result.Store(&out)

		for i := 0; i < 4; i++ {
			expected := MRedLazy(xArr[i], yArr[i], Q, MRedConst)
			if out[i] != expected {
				t.Errorf("mredLazy4 lane %d: x=%d, y=%d, got %d, want %d",
					i, xArr[i], yArr[i], out[i], expected)
			}
		}
	}
}

// TestMul64x4 verifies the SIMD 64x64->128 multiply against bits.Mul64.
func TestMul64x4(t *testing.T) {
	if !simdAVX2 {
		t.Skip("AVX2 not available")
	}

	testPairs := [][2][4]uint64{
		{{0, 1, 2, 3}, {0, 1, 2, 3}},
		{{0xFFFFFFFF, 0x100000000, 0xFFFFFFFFFFFFFFFF, 1}, {0xFFFFFFFF, 0x100000000, 0xFFFFFFFFFFFFFFFF, 1}},
		{{0xDEADBEEFCAFEBABE, 123456789, 0, 1 << 63}, {0xCAFEBABEDEADBEEF, 987654321, 42, 1 << 63}},
		{{576460752303439873, 576460752303439872, 1<<62 + 1, 1<<61 - 1}, {576460752303702017, 576460752303702016, 1<<62 - 1, 1<<61 + 1}},
	}

	for _, pair := range testPairs {
		aArr := pair[0]
		bArr := pair[1]

		hi, lo := mul64x4(
			archsimd.LoadUint64x4(&aArr),
			archsimd.LoadUint64x4(&bArr),
		)

		var hiOut, loOut [4]uint64
		hi.Store(&hiOut)
		lo.Store(&loOut)

		for i := 0; i < 4; i++ {
			expectHi, expectLo := bits.Mul64(aArr[i], bArr[i])
			if hiOut[i] != expectHi || loOut[i] != expectLo {
				t.Errorf("mul64x4 lane %d: a=%d, b=%d, got hi=%d lo=%d, want hi=%d lo=%d",
					i, aArr[i], bArr[i], hiOut[i], loOut[i], expectHi, expectLo)
			}
		}
	}
}

// TestNTTSimdMatchesScalar verifies that SIMD NTT produces identical output to scalar NTT.
func TestNTTSimdMatchesScalar(t *testing.T) {
	if !simdAVX2 {
		t.Skip("AVX2 not available")
	}

	for _, tv := range testVector[:] {
		ringQ, err := NewRing(tv.N, tv.Qis)
		if err != nil {
			t.Fatal(err)
		}

		t.Run("NTT/SIMD", func(t *testing.T) {
			x := ringQ.NewPoly()
			x.Copy(tv.poly)

			simdResult := ringQ.NewPoly()
			ringQ.NTT(x, simdResult)

			// The SIMD path should produce the same result as the known test vector
			if !ringQ.Equal(simdResult, tv.polyNTT) {
				t.Error("SIMD NTT result does not match expected test vector")
			}
		})

		t.Run("INTT/SIMD", func(t *testing.T) {
			x := ringQ.NewPoly()
			x.Copy(tv.polyNTT)

			result := ringQ.NewPoly()
			ringQ.INTT(x, result)

			if !ringQ.Equal(result, tv.poly) {
				t.Error("SIMD INTT result does not match expected test vector")
			}
		})
	}
}
