//go:build !cgo || !gpu

// Package gpu provides pure Go implementations when CGO is disabled or the gpu build tag is not set.
// These implementations use the lattice/ring package for NTT and polynomial operations.
package gpu

import (
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"sync"

	"github.com/luxfi/lattice/v7/ring"
	"github.com/luxfi/lattice/v7/utils/sampling"
)

// GPUAvailable returns false when CGO is disabled.
func GPUAvailable() bool {
	return false
}

// GetBackend returns "CPU (pure Go)".
func GetBackend() string {
	return "CPU (pure Go)"
}

// ClearCache is a no-op without GPU.
func ClearCache() {}

// NTTContext wraps the ring.SubRing for NTT operations.
type NTTContext struct {
	subRing *ring.SubRing
	N       uint32
	Q       uint64
	mu      sync.RWMutex
}

// NewNTTContext creates a new NTT context using pure Go implementation.
func NewNTTContext(N uint32, Q uint64) (*NTTContext, error) {
	if N == 0 || (N&(N-1)) != 0 {
		return nil, fmt.Errorf("N must be a power of 2, got %d", N)
	}

	// Check if Q is NTT-friendly (Q-1 divisible by 2N)
	if (Q-1)%(2*uint64(N)) != 0 {
		return nil, fmt.Errorf("Q-1 (%d) must be divisible by 2N (%d) for NTT-friendly prime", Q-1, 2*uint64(N))
	}

	subRing, err := ring.NewSubRing(int(N), Q)
	if err != nil {
		return nil, fmt.Errorf("failed to create SubRing: %w", err)
	}

	// NTT constants are generated internally by NewSubRing via NewSubRingWithCustomNTT

	return &NTTContext{
		subRing: subRing,
		N:       N,
		Q:       Q,
	}, nil
}

// Close releases resources (no-op for pure Go).
func (ctx *NTTContext) Close() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	ctx.subRing = nil
}

// NTT performs forward NTT on a batch of polynomials.
func (ctx *NTTContext) NTT(polys [][]uint64) ([][]uint64, error) {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.subRing == nil {
		return nil, fmt.Errorf("NTT context is closed")
	}

	if len(polys) == 0 {
		return polys, nil
	}

	N := int(ctx.N)
	results := make([][]uint64, len(polys))

	for i, poly := range polys {
		if len(poly) != N {
			return nil, fmt.Errorf("polynomial %d has wrong size: got %d, expected %d", i, len(poly), N)
		}

		// Copy to result
		results[i] = make([]uint64, N)
		copy(results[i], poly)

		// Perform NTT in-place
		ctx.subRing.NTT(results[i], results[i])
	}

	return results, nil
}

// INTT performs inverse NTT on a batch of polynomials.
func (ctx *NTTContext) INTT(polys [][]uint64) ([][]uint64, error) {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.subRing == nil {
		return nil, fmt.Errorf("NTT context is closed")
	}

	if len(polys) == 0 {
		return polys, nil
	}

	N := int(ctx.N)
	results := make([][]uint64, len(polys))

	for i, poly := range polys {
		if len(poly) != N {
			return nil, fmt.Errorf("polynomial %d has wrong size: got %d, expected %d", i, len(poly), N)
		}

		// Copy to result
		results[i] = make([]uint64, N)
		copy(results[i], poly)

		// Perform INTT in-place
		ctx.subRing.INTT(results[i], results[i])
	}

	return results, nil
}

// PolyMul performs polynomial multiplication using pure Go NTT.
func (ctx *NTTContext) PolyMul(a, b [][]uint64) ([][]uint64, error) {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.subRing == nil {
		return nil, fmt.Errorf("NTT context is closed")
	}

	if len(a) != len(b) {
		return nil, fmt.Errorf("batch size mismatch: %d vs %d", len(a), len(b))
	}

	if len(a) == 0 {
		return nil, nil
	}

	N := int(ctx.N)
	results := make([][]uint64, len(a))
	tempA := make([]uint64, N)
	tempB := make([]uint64, N)

	for i := range a {
		if len(a[i]) != N || len(b[i]) != N {
			return nil, fmt.Errorf("polynomial %d has wrong size", i)
		}

		results[i] = make([]uint64, N)

		// Copy inputs
		copy(tempA, a[i])
		copy(tempB, b[i])

		// Forward NTT
		ctx.subRing.NTT(tempA, tempA)
		ctx.subRing.NTT(tempB, tempB)

		// Pointwise multiply
		ctx.subRing.MulCoeffsMontgomery(tempA, tempB, results[i])

		// Inverse NTT
		ctx.subRing.INTT(results[i], results[i])
	}

	return results, nil
}

// PolyMulNTT performs element-wise multiplication in NTT domain.
func (ctx *NTTContext) PolyMulNTT(a, b []uint64) ([]uint64, error) {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	if ctx.subRing == nil {
		return nil, fmt.Errorf("NTT context is closed")
	}

	N := int(ctx.N)
	if len(a) != N || len(b) != N {
		return nil, fmt.Errorf("polynomial size mismatch")
	}

	result := make([]uint64, N)
	ctx.subRing.MulCoeffsMontgomery(a, b, result)

	return result, nil
}

// PolyAdd computes result = a + b (mod Q).
func PolyAdd(a, b []uint64, Q uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("polynomial size mismatch")
	}

	result := make([]uint64, len(a))
	for i := range a {
		sum := a[i] + b[i]
		if sum >= Q {
			sum -= Q
		}
		result[i] = sum
	}

	return result, nil
}

// PolySub computes result = a - b (mod Q).
func PolySub(a, b []uint64, Q uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("polynomial size mismatch")
	}

	result := make([]uint64, len(a))
	for i := range a {
		if a[i] >= b[i] {
			result[i] = a[i] - b[i]
		} else {
			result[i] = Q - b[i] + a[i]
		}
	}

	return result, nil
}

// PolyScalarMul computes result = a * scalar (mod Q).
func PolyScalarMul(a []uint64, scalar, Q uint64) ([]uint64, error) {
	result := make([]uint64, len(a))
	for i := range a {
		// Use big.Int for 128-bit intermediate
		prod := new(big.Int).SetUint64(a[i])
		prod.Mul(prod, new(big.Int).SetUint64(scalar))
		prod.Mod(prod, new(big.Int).SetUint64(Q))
		result[i] = prod.Uint64()
	}
	return result, nil
}

// SampleGaussian samples a polynomial with discrete Gaussian distribution.
func SampleGaussian(N uint32, Q uint64, sigma float64, seed []byte) ([]uint64, error) {
	result := make([]uint64, N)

	// Create PRNG
	var prng sampling.PRNG
	if len(seed) >= 32 {
		prng, _ = sampling.NewKeyedPRNG(seed[:32])
	} else {
		prng, _ = sampling.NewPRNG()
	}

	// Use Box-Muller transform for Gaussian sampling
	bound := int64(math.Ceil(sigma * 6)) // 6-sigma cutoff

	for i := uint32(0); i < N; i++ {
		// Rejection sampling for discrete Gaussian
		for {
			// Sample from bounded uniform
			var sample int64
			for {
				b := make([]byte, 8)
				prng.Read(b)
				sample = int64(b[0]) | int64(b[1])<<8 | int64(b[2])<<16 | int64(b[3])<<24
				sample = sample % (2*bound + 1) - bound
				if sample >= -bound && sample <= bound {
					break
				}
			}

			// Accept with Gaussian probability
			prob := math.Exp(-float64(sample*sample) / (2 * sigma * sigma))
			threshold := make([]byte, 8)
			prng.Read(threshold)
			if float64(threshold[0])/256.0 < prob {
				if sample >= 0 {
					result[i] = uint64(sample)
				} else {
					result[i] = Q - uint64(-sample)
				}
				break
			}
		}
	}

	return result, nil
}

// SampleUniform samples a uniform random polynomial.
func SampleUniform(N uint32, Q uint64, seed []byte) ([]uint64, error) {
	result := make([]uint64, N)

	// Create PRNG
	var prng sampling.PRNG
	if len(seed) >= 32 {
		prng, _ = sampling.NewKeyedPRNG(seed[:32])
	} else {
		prng, _ = sampling.NewPRNG()
	}

	qBig := new(big.Int).SetUint64(Q)

	for i := uint32(0); i < N; i++ {
		// Sample uniform in [0, Q)
		b := make([]byte, 8)
		prng.Read(b)
		val := new(big.Int).SetBytes(b)
		val.Mod(val, qBig)
		result[i] = val.Uint64()
	}

	return result, nil
}

// SampleTernary samples a ternary polynomial {-1, 0, 1}.
func SampleTernary(N uint32, Q uint64, density float64, seed []byte) ([]uint64, error) {
	result := make([]uint64, N)

	// Create PRNG
	var reader interface{ Read([]byte) (int, error) }
	if len(seed) >= 32 {
		prng, _ := sampling.NewKeyedPRNG(seed[:32])
		reader = prng
	} else {
		reader = rand.Reader
	}

	for i := uint32(0); i < N; i++ {
		b := make([]byte, 2)
		reader.Read(b)

		// Probability of non-zero
		p := float64(b[0]) / 256.0
		if p < density {
			// Non-zero: choose -1 or 1
			if b[1]&1 == 0 {
				result[i] = 1
			} else {
				result[i] = Q - 1 // -1 mod Q
			}
		} else {
			result[i] = 0
		}
	}

	return result, nil
}

// FindPrimitiveRoot finds a primitive 2N-th root of unity modulo Q.
func FindPrimitiveRoot(N uint32, Q uint64) (uint64, error) {
	// Compute g = generator of Z_Q^*
	// Then root = g^((Q-1)/(2N))
	exponent := (Q - 1) / (2 * uint64(N))

	// Find a generator (primitive root of Q)
	// For prime Q, try small values until we find one
	for g := uint64(2); g < Q; g++ {
		// Check if g is a primitive root
		gBig := new(big.Int).SetUint64(g)
		qBig := new(big.Int).SetUint64(Q)
		expBig := new(big.Int).SetUint64(exponent)

		root := new(big.Int).Exp(gBig, expBig, qBig)

		// Verify it's a 2N-th root of unity
		twoN := new(big.Int).SetUint64(2 * uint64(N))
		test := new(big.Int).Exp(root, twoN, qBig)
		if test.Uint64() == 1 {
			// Also verify root^N != 1 (primitive)
			nBig := new(big.Int).SetUint64(uint64(N))
			testN := new(big.Int).Exp(root, nBig, qBig)
			if testN.Uint64() != 1 {
				return root.Uint64(), nil
			}
		}
	}

	return 0, fmt.Errorf("no primitive root found for N=%d, Q=%d", N, Q)
}

// ModInverse computes the modular inverse a^{-1} mod Q.
func ModInverse(a, Q uint64) (uint64, error) {
	aBig := new(big.Int).SetUint64(a)
	qBig := new(big.Int).SetUint64(Q)
	inv := new(big.Int).ModInverse(aBig, qBig)
	if inv == nil {
		return 0, fmt.Errorf("%d is not invertible mod %d", a, Q)
	}
	return inv.Uint64(), nil
}

// IsNTTPrime checks if Q is a valid NTT-friendly prime for ring dimension N.
func IsNTTPrime(N uint32, Q uint64) bool {
	// Q must be prime and Q ≡ 1 (mod 2N)
	if (Q-1)%(2*uint64(N)) != 0 {
		return false
	}

	// Check if Q is prime (simple Miller-Rabin)
	qBig := new(big.Int).SetUint64(Q)
	return qBig.ProbablyPrime(20)
}
