//go:build cgo

package gpu

import (
	"testing"
)

// Test NTT-friendly prime used by Ringtail
const (
	TestN uint32 = 256
	TestQ uint64 = 0x1000000004A01 // 48-bit NTT-friendly prime
)

func TestGPUAvailable(t *testing.T) {
	available := GPUAvailable()
	backend := GetBackend()
	t.Logf("GPU Available: %v, Backend: %s", available, backend)
}

func TestIsNTTPrime(t *testing.T) {
	// Test Ringtail primes
	tests := []struct {
		N    uint32
		Q    uint64
		want bool
		name string
	}{
		{256, 0x1000000004A01, true, "Ringtail Q"},
		{256, 0x40201, true, "Ringtail QXi"},   // 262657 - prime, ≡ 1 (mod 512)
		{256, 0x7FE01, true, "Ringtail QNu"},   // 523777 - prime, ≡ 1 (mod 512)
		{256, 0x40000, false, "Non-prime 2^18"}, // 262144 - NOT prime
		{256, 0x80000, false, "Non-prime 2^19"}, // 524288 - NOT prime
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsNTTPrime(tt.N, tt.Q)
			if got != tt.want {
				t.Errorf("IsNTTPrime(%d, 0x%X) = %v, want %v", tt.N, tt.Q, got, tt.want)
			}
		})
	}
}

func TestNewNTTContext(t *testing.T) {
	ctx, err := NewNTTContext(TestN, TestQ)
	if err != nil {
		t.Fatalf("NewNTTContext failed: %v", err)
	}
	defer ctx.Close()

	if ctx.N != TestN {
		t.Errorf("N = %d, want %d", ctx.N, TestN)
	}
	if ctx.Q != TestQ {
		t.Errorf("Q = 0x%X, want 0x%X", ctx.Q, TestQ)
	}
}

func TestNTTRoundTrip(t *testing.T) {
	ctx, err := NewNTTContext(TestN, TestQ)
	if err != nil {
		t.Fatalf("NewNTTContext failed: %v", err)
	}
	defer ctx.Close()

	// Create a test polynomial
	poly := make([]uint64, TestN)
	for i := range poly {
		poly[i] = uint64(i + 1) % TestQ
	}

	// Save original
	original := make([]uint64, TestN)
	copy(original, poly)

	// Forward NTT
	nttResult, err := ctx.NTT([][]uint64{poly})
	if err != nil {
		t.Fatalf("NTT failed: %v", err)
	}

	// Inverse NTT
	inttResult, err := ctx.INTT(nttResult)
	if err != nil {
		t.Fatalf("INTT failed: %v", err)
	}

	// Verify round-trip
	for i := range original {
		if inttResult[0][i] != original[i] {
			t.Errorf("Round-trip failed at index %d: got %d, want %d", i, inttResult[0][i], original[i])
		}
	}

	t.Log("NTT round-trip successful")
}

func TestFindPrimitiveRoot(t *testing.T) {
	root, err := FindPrimitiveRoot(TestN, TestQ)
	if err != nil {
		t.Fatalf("FindPrimitiveRoot failed: %v", err)
	}
	t.Logf("Primitive 2×%d-th root of unity mod 0x%X: %d", TestN, TestQ, root)

	if root == 0 || root >= TestQ {
		t.Errorf("Invalid root: %d", root)
	}
}

func TestModInverse(t *testing.T) {
	// Test N^{-1} mod Q
	inv, err := ModInverse(uint64(TestN), TestQ)
	if err != nil {
		t.Fatalf("ModInverse failed: %v", err)
	}

	// Verify: N * N^{-1} = 1 (mod Q)
	product := (uint64(TestN) * inv) % TestQ
	if product != 1 {
		t.Errorf("ModInverse verification failed: %d * %d = %d (mod Q), want 1", TestN, inv, product)
	}
	t.Logf("%d^{-1} mod 0x%X = %d", TestN, TestQ, inv)
}

func TestPolyArithmetic(t *testing.T) {
	N := uint32(TestN)
	Q := TestQ

	a := make([]uint64, N)
	b := make([]uint64, N)
	for i := range a {
		a[i] = uint64(i+1) % Q
		b[i] = uint64(i+2) % Q
	}

	// Test addition
	sum, err := PolyAdd(a, b, Q)
	if err != nil {
		t.Fatalf("PolyAdd failed: %v", err)
	}
	for i := range sum {
		expected := (a[i] + b[i]) % Q
		if sum[i] != expected {
			t.Errorf("PolyAdd[%d] = %d, want %d", i, sum[i], expected)
		}
	}

	// Test subtraction
	diff, err := PolySub(a, b, Q)
	if err != nil {
		t.Fatalf("PolySub failed: %v", err)
	}
	for i := range diff {
		var expected uint64
		if a[i] >= b[i] {
			expected = a[i] - b[i]
		} else {
			expected = Q - b[i] + a[i]
		}
		if diff[i] != expected {
			t.Errorf("PolySub[%d] = %d, want %d", i, diff[i], expected)
		}
	}

	// Test scalar multiplication
	scalar := uint64(5)
	product, err := PolyScalarMul(a, scalar, Q)
	if err != nil {
		t.Fatalf("PolyScalarMul failed: %v", err)
	}
	for i := range product {
		expected := (a[i] * scalar) % Q
		if product[i] != expected {
			t.Errorf("PolyScalarMul[%d] = %d, want %d", i, product[i], expected)
		}
	}

	t.Log("Polynomial arithmetic tests passed")
}

func BenchmarkNTT(b *testing.B) {
	ctx, err := NewNTTContext(TestN, TestQ)
	if err != nil {
		b.Fatalf("NewNTTContext failed: %v", err)
	}
	defer ctx.Close()

	poly := make([]uint64, TestN)
	for i := range poly {
		poly[i] = uint64(i) % TestQ
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ctx.NTT([][]uint64{poly})
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPolyMul(b *testing.B) {
	ctx, err := NewNTTContext(TestN, TestQ)
	if err != nil {
		b.Fatalf("NewNTTContext failed: %v", err)
	}
	defer ctx.Close()

	a := make([]uint64, TestN)
	c := make([]uint64, TestN)
	for i := range a {
		a[i] = uint64(i) % TestQ
		c[i] = uint64(i+1) % TestQ
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ctx.PolyMul([][]uint64{a}, [][]uint64{c})
		if err != nil {
			b.Fatal(err)
		}
	}
}
