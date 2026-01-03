//go:build cgo && gpu

// Package gpu provides GPU-accelerated lattice operations via libLattice.
//
// With CGO enabled, this package links to libLattice for GPU acceleration:
//   - Metal (macOS/Apple Silicon)
//   - CUDA (Linux/NVIDIA)
//   - Optimized CPU fallback
//
// The library automatically selects the best available backend.
//
// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
package gpu

/*
#cgo pkg-config: lux-lattice
#cgo CXXFLAGS: -std=c++17 -O3
#cgo darwin LDFLAGS: -framework Metal -framework Foundation -lstdc++
#cgo linux LDFLAGS: -lstdc++ -lcudart

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// libLattice GPU-accelerated operations
bool lattice_gpu_available(void);
const char* lattice_get_backend(void);
void lattice_clear_cache(void);

// NTT context
typedef struct LatticeNTTContext LatticeNTTContext;
LatticeNTTContext* lattice_ntt_create(uint32_t N, uint64_t Q);
void lattice_ntt_destroy(LatticeNTTContext* ctx);
void lattice_ntt_get_params(const LatticeNTTContext* ctx, uint32_t* N, uint64_t* Q);

// NTT operations
int lattice_ntt_forward(LatticeNTTContext* ctx, uint64_t* data, uint32_t batch);
int lattice_ntt_inverse(LatticeNTTContext* ctx, uint64_t* data, uint32_t batch);

// Polynomial operations
int lattice_poly_mul_ntt(LatticeNTTContext* ctx, uint64_t* result, const uint64_t* a, const uint64_t* b);
int lattice_poly_mul(LatticeNTTContext* ctx, uint64_t* result, const uint64_t* a, const uint64_t* b);
int lattice_poly_add(uint64_t* result, const uint64_t* a, const uint64_t* b, uint32_t N, uint64_t Q);
int lattice_poly_sub(uint64_t* result, const uint64_t* a, const uint64_t* b, uint32_t N, uint64_t Q);
int lattice_poly_scalar_mul(uint64_t* result, const uint64_t* a, uint64_t scalar, uint32_t N, uint64_t Q);

// Sampling
int lattice_sample_gaussian(uint64_t* result, uint32_t N, uint64_t Q, double sigma, const uint8_t* seed);
int lattice_sample_uniform(uint64_t* result, uint32_t N, uint64_t Q, const uint8_t* seed);
int lattice_sample_ternary(uint64_t* result, uint32_t N, uint64_t Q, double density, const uint8_t* seed);

// Utility
uint64_t lattice_find_primitive_root(uint32_t N, uint64_t Q);
uint64_t lattice_mod_inverse(uint64_t a, uint64_t Q);
bool lattice_is_ntt_prime(uint32_t N, uint64_t Q);
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// GPUAvailable returns true if GPU acceleration is available.
func GPUAvailable() bool {
	return bool(C.lattice_gpu_available())
}

// GetBackend returns the name of the active backend ("Metal", "CUDA", or "CPU").
func GetBackend() string {
	return C.GoString(C.lattice_get_backend())
}

// ClearCache clears internal caches (twiddle factors, contexts).
func ClearCache() {
	C.lattice_clear_cache()
}

// NTTContext holds precomputed data for GPU-accelerated NTT operations.
type NTTContext struct {
	ptr *C.LatticeNTTContext
	N   uint32
	Q   uint64
	mu  sync.RWMutex
}

// NewNTTContext creates a new NTT context for the given ring parameters.
// N must be a power of 2, Q must be an NTT-friendly prime (Q ≡ 1 mod 2N).
func NewNTTContext(N uint32, Q uint64) (*NTTContext, error) {
	if N == 0 || (N&(N-1)) != 0 {
		return nil, fmt.Errorf("N must be a power of 2, got %d", N)
	}

	if (Q-1)%(2*uint64(N)) != 0 {
		return nil, fmt.Errorf("Q-1 (%d) must be divisible by 2N (%d) for NTT-friendly prime", Q-1, 2*uint64(N))
	}

	ptr := C.lattice_ntt_create(C.uint32_t(N), C.uint64_t(Q))
	if ptr == nil {
		return nil, fmt.Errorf("failed to create NTT context for N=%d, Q=%d", N, Q)
	}

	return &NTTContext{
		ptr: ptr,
		N:   N,
		Q:   Q,
	}, nil
}

// Close releases the NTT context resources.
func (ctx *NTTContext) Close() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	if ctx.ptr != nil {
		C.lattice_ntt_destroy(ctx.ptr)
		ctx.ptr = nil
	}
}

// NTT performs forward NTT on a batch of polynomials.
// Each polynomial is transformed in-place from coefficient to NTT domain.
func (ctx *NTTContext) NTT(polys [][]uint64) ([][]uint64, error) {
	if ctx.ptr == nil {
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

		// Copy to result (NTT modifies in-place)
		results[i] = make([]uint64, N)
		copy(results[i], poly)

		err := C.lattice_ntt_forward(ctx.ptr, (*C.uint64_t)(unsafe.Pointer(&results[i][0])), 1)
		if err != 0 {
			return nil, fmt.Errorf("NTT forward failed with error %d", err)
		}
	}

	return results, nil
}

// INTT performs inverse NTT on a batch of polynomials.
// Each polynomial is transformed in-place from NTT to coefficient domain.
func (ctx *NTTContext) INTT(polys [][]uint64) ([][]uint64, error) {
	if ctx.ptr == nil {
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

		// Copy to result (INTT modifies in-place)
		results[i] = make([]uint64, N)
		copy(results[i], poly)

		err := C.lattice_ntt_inverse(ctx.ptr, (*C.uint64_t)(unsafe.Pointer(&results[i][0])), 1)
		if err != 0 {
			return nil, fmt.Errorf("INTT inverse failed with error %d", err)
		}
	}

	return results, nil
}

// PolyMul performs polynomial multiplication using GPU-accelerated NTT.
// Both polynomials should be in coefficient form.
// Returns a * b in R_Q = Z_Q[X]/(X^N + 1).
func (ctx *NTTContext) PolyMul(a, b [][]uint64) ([][]uint64, error) {
	if ctx.ptr == nil {
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

	for i := range a {
		if len(a[i]) != N || len(b[i]) != N {
			return nil, fmt.Errorf("polynomial %d has wrong size", i)
		}

		results[i] = make([]uint64, N)

		err := C.lattice_poly_mul(ctx.ptr,
			(*C.uint64_t)(unsafe.Pointer(&results[i][0])),
			(*C.uint64_t)(unsafe.Pointer(&a[i][0])),
			(*C.uint64_t)(unsafe.Pointer(&b[i][0])))
		if err != 0 {
			return nil, fmt.Errorf("polynomial multiplication failed with error %d", err)
		}
	}

	return results, nil
}

// PolyMulNTT performs element-wise multiplication (Hadamard product) in NTT domain.
// Both polynomials must already be in NTT form.
func (ctx *NTTContext) PolyMulNTT(a, b []uint64) ([]uint64, error) {
	if ctx.ptr == nil {
		return nil, fmt.Errorf("NTT context is closed")
	}

	N := int(ctx.N)
	if len(a) != N || len(b) != N {
		return nil, fmt.Errorf("polynomial size mismatch")
	}

	result := make([]uint64, N)

	err := C.lattice_poly_mul_ntt(ctx.ptr,
		(*C.uint64_t)(unsafe.Pointer(&result[0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])))
	if err != 0 {
		return nil, fmt.Errorf("NTT multiplication failed with error %d", err)
	}

	return result, nil
}

// PolyAdd computes result = a + b (mod Q).
func PolyAdd(a, b []uint64, Q uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("polynomial size mismatch")
	}

	N := uint32(len(a))
	result := make([]uint64, N)

	err := C.lattice_poly_add(
		(*C.uint64_t)(unsafe.Pointer(&result[0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		C.uint32_t(N),
		C.uint64_t(Q))
	if err != 0 {
		return nil, fmt.Errorf("polynomial addition failed with error %d", err)
	}

	return result, nil
}

// PolySub computes result = a - b (mod Q).
func PolySub(a, b []uint64, Q uint64) ([]uint64, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("polynomial size mismatch")
	}

	N := uint32(len(a))
	result := make([]uint64, N)

	err := C.lattice_poly_sub(
		(*C.uint64_t)(unsafe.Pointer(&result[0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		C.uint32_t(N),
		C.uint64_t(Q))
	if err != 0 {
		return nil, fmt.Errorf("polynomial subtraction failed with error %d", err)
	}

	return result, nil
}

// PolyScalarMul computes result = a * scalar (mod Q).
func PolyScalarMul(a []uint64, scalar, Q uint64) ([]uint64, error) {
	N := uint32(len(a))
	result := make([]uint64, N)

	err := C.lattice_poly_scalar_mul(
		(*C.uint64_t)(unsafe.Pointer(&result[0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0])),
		C.uint64_t(scalar),
		C.uint32_t(N),
		C.uint64_t(Q))
	if err != 0 {
		return nil, fmt.Errorf("scalar multiplication failed with error %d", err)
	}

	return result, nil
}

// SampleGaussian samples a polynomial with discrete Gaussian distribution.
func SampleGaussian(N uint32, Q uint64, sigma float64, seed []byte) ([]uint64, error) {
	result := make([]uint64, N)

	var seedPtr *C.uint8_t
	if len(seed) >= 32 {
		seedPtr = (*C.uint8_t)(unsafe.Pointer(&seed[0]))
	}

	err := C.lattice_sample_gaussian(
		(*C.uint64_t)(unsafe.Pointer(&result[0])),
		C.uint32_t(N),
		C.uint64_t(Q),
		C.double(sigma),
		seedPtr)
	if err != 0 {
		return nil, fmt.Errorf("Gaussian sampling failed with error %d", err)
	}

	return result, nil
}

// SampleUniform samples a uniform random polynomial.
func SampleUniform(N uint32, Q uint64, seed []byte) ([]uint64, error) {
	result := make([]uint64, N)

	var seedPtr *C.uint8_t
	if len(seed) >= 32 {
		seedPtr = (*C.uint8_t)(unsafe.Pointer(&seed[0]))
	}

	err := C.lattice_sample_uniform(
		(*C.uint64_t)(unsafe.Pointer(&result[0])),
		C.uint32_t(N),
		C.uint64_t(Q),
		seedPtr)
	if err != 0 {
		return nil, fmt.Errorf("uniform sampling failed with error %d", err)
	}

	return result, nil
}

// SampleTernary samples a ternary polynomial {-1, 0, 1}.
func SampleTernary(N uint32, Q uint64, density float64, seed []byte) ([]uint64, error) {
	result := make([]uint64, N)

	var seedPtr *C.uint8_t
	if len(seed) >= 32 {
		seedPtr = (*C.uint8_t)(unsafe.Pointer(&seed[0]))
	}

	err := C.lattice_sample_ternary(
		(*C.uint64_t)(unsafe.Pointer(&result[0])),
		C.uint32_t(N),
		C.uint64_t(Q),
		C.double(density),
		seedPtr)
	if err != 0 {
		return nil, fmt.Errorf("ternary sampling failed with error %d", err)
	}

	return result, nil
}

// FindPrimitiveRoot finds a primitive 2N-th root of unity modulo Q.
func FindPrimitiveRoot(N uint32, Q uint64) (uint64, error) {
	root := uint64(C.lattice_find_primitive_root(C.uint32_t(N), C.uint64_t(Q)))
	if root == 0 {
		return 0, fmt.Errorf("no primitive root found for N=%d, Q=%d", N, Q)
	}
	return root, nil
}

// ModInverse computes the modular inverse a^{-1} mod Q.
func ModInverse(a, Q uint64) (uint64, error) {
	inv := uint64(C.lattice_mod_inverse(C.uint64_t(a), C.uint64_t(Q)))
	if inv == 0 && a != 1 {
		return 0, fmt.Errorf("%d is not invertible mod %d", a, Q)
	}
	return inv, nil
}

// IsNTTPrime checks if Q is a valid NTT-friendly prime for ring dimension N.
func IsNTTPrime(N uint32, Q uint64) bool {
	return bool(C.lattice_is_ntt_prime(C.uint32_t(N), C.uint64_t(Q)))
}
