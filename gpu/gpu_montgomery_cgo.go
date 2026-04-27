//go:build cgo && gpu

// Montgomery-form GPU dispatch.
//
// This file exposes the parts of the GPU NTT that match the Lattigo
// contract: SubRing.RootsForward / RootsBackward / NInv / MRedConstant
// flow through unchanged. Output of forward/inverse is byte-equal to
// ring.NTTStandard / ring.INTTStandard for the same SubRing.
//
// The Ring.NTT fast-path (subring_ops.go) consults Available() and
// SubRingNTTForward/Backward to opt into GPU dispatch.
//
// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
package gpu

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct LatticeNTTContext LatticeNTTContext;

LatticeNTTContext* lattice_ntt_create_montgomery(uint32_t N, uint64_t Q,
                                                 uint64_t mred_constant,
                                                 uint64_t n_inv,
                                                 const uint64_t* roots_forward,
                                                 const uint64_t* roots_backward);
void lattice_ntt_destroy(LatticeNTTContext* ctx);
int  lattice_ntt_forward(LatticeNTTContext* ctx, uint64_t* data, uint32_t batch);
int  lattice_ntt_inverse(LatticeNTTContext* ctx, uint64_t* data, uint32_t batch);
*/
import "C"

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/luxfi/lattice/v7/ring"
)

// MontgomeryNTTContext binds a luxfi/lattice/v7/ring.SubRing to a
// GPU-resident NTT context. The output of Forward and Backward is
// byte-equal to the pure-Go ring.SubRing.NTT and INTT for the same
// inputs.
type MontgomeryNTTContext struct {
	ptr *C.LatticeNTTContext
	N   uint32
	Q   uint64
	mu  sync.RWMutex
}

// NewMontgomeryNTTContext binds to a SubRing. Captures Montgomery roots
// at construction time -- subsequent SubRing mutations are not seen.
func NewMontgomeryNTTContext(s *ring.SubRing) (*MontgomeryNTTContext, error) {
	if s == nil {
		return nil, fmt.Errorf("nil SubRing")
	}
	if s.NTTTable == nil {
		return nil, fmt.Errorf("SubRing.NTTTable is nil -- call ring.NewRing first")
	}
	if s.N <= 0 {
		return nil, fmt.Errorf("invalid SubRing.N=%d", s.N)
	}
	if len(s.RootsForward) < s.N || len(s.RootsBackward) < s.N {
		return nil, fmt.Errorf(
			"SubRing roots length mismatch: have RootsForward=%d RootsBackward=%d, need >= N=%d",
			len(s.RootsForward), len(s.RootsBackward), s.N)
	}

	N := uint32(s.N)
	Q := s.Modulus

	// The ring stores bit-reversed roots in arrays of length N (NthRoot/2
	// where NthRoot = 2N for the Standard ring). The C ABI expects a
	// length-N array indexed exactly the same way the butterfly loop
	// indexes it: `roots[m+i]` where `m+i` ranges over [1, N).
	rfwd := s.RootsForward[:N]
	rbwd := s.RootsBackward[:N]

	ptr := C.lattice_ntt_create_montgomery(
		C.uint32_t(N),
		C.uint64_t(Q),
		C.uint64_t(s.MRedConstant),
		C.uint64_t(s.NInv),
		(*C.uint64_t)(unsafe.Pointer(&rfwd[0])),
		(*C.uint64_t)(unsafe.Pointer(&rbwd[0])),
	)
	if ptr == nil {
		return nil, fmt.Errorf("lattice_ntt_create_montgomery failed for N=%d Q=%d", N, Q)
	}

	return &MontgomeryNTTContext{ptr: ptr, N: N, Q: Q}, nil
}

// Close releases the GPU context.
func (ctx *MontgomeryNTTContext) Close() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.ptr != nil {
		C.lattice_ntt_destroy(ctx.ptr)
		ctx.ptr = nil
	}
}

// Forward in-place. data must have length batch*N.
func (ctx *MontgomeryNTTContext) Forward(data []uint64, batch uint32) error {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.ptr == nil {
		return fmt.Errorf("context closed")
	}
	if int(batch)*int(ctx.N) > len(data) {
		return fmt.Errorf("buffer too small: need %d, got %d",
			int(batch)*int(ctx.N), len(data))
	}
	rc := C.lattice_ntt_forward(ctx.ptr, (*C.uint64_t)(unsafe.Pointer(&data[0])), C.uint32_t(batch))
	if rc != 0 {
		return fmt.Errorf("lattice_ntt_forward returned %d", int(rc))
	}
	return nil
}

// Backward in-place.
func (ctx *MontgomeryNTTContext) Backward(data []uint64, batch uint32) error {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.ptr == nil {
		return fmt.Errorf("context closed")
	}
	if int(batch)*int(ctx.N) > len(data) {
		return fmt.Errorf("buffer too small: need %d, got %d",
			int(batch)*int(ctx.N), len(data))
	}
	rc := C.lattice_ntt_inverse(ctx.ptr, (*C.uint64_t)(unsafe.Pointer(&data[0])), C.uint32_t(batch))
	if rc != 0 {
		return fmt.Errorf("lattice_ntt_inverse returned %d", int(rc))
	}
	return nil
}

// =============================================================================
// SubRing dispatch fast-path
// =============================================================================
//
// Available() reports whether the GPU NTT is reachable on this build (cgo
// + GPU library + Metal/CUDA at runtime). NTTThreshold is the per-poly
// crossover: below this N, Go pure-Go always wins on this backend (Metal
// command-buffer overhead floor measured at ~12-15 µs vs <1 µs Go). At
// or above this N the Metal path is competitive on M1 Max; for batch
// dispatch the threshold drops further but is gated by the caller (the
// ring core only ever calls per-SubRing single-poly NTT).

// NTTThreshold is the minimum N for which a single-poly GPU dispatch is
// considered worthwhile. Tunable; based on direct measurement on M1 Max
// with the Montgomery kernel after the BatchNTT fix landed. Override at
// runtime via SetNTTThreshold.
var nttThreshold atomic.Uint32

func init() {
	// 0 means "GPU dispatch disabled by default for single-poly". The
	// fast-path remains opt-in until a caller explicitly registers a
	// MontgomeryNTTContext for a given SubRing -- see RegisterSubRing.
	// Single-poly Metal NTT is strictly slower than Go on M1 Max for
	// every measured N up to 16384, so we do NOT auto-dispatch.
	nttThreshold.Store(0)

	// Register the dispatchers with the ring package so SubRing.NTT and
	// SubRing.INTT consult our fast-path. Pure-Go behaviour is preserved
	// for any SubRing that has not been explicitly registered.
	ring.SetGPUDispatchers(NTTSubRingForward, NTTSubRingBackward)
}

// SetNTTThreshold overrides the single-poly dispatch threshold. Pass 0
// to disable single-poly GPU dispatch entirely (still allows batched
// dispatch via NewMontgomeryNTTContext directly). Pass math.MaxUint32 to
// dispatch every NTT through GPU (test mode).
func SetNTTThreshold(n uint32) { nttThreshold.Store(n) }

// NTTThreshold returns the current threshold.
func NTTThreshold() uint32 { return nttThreshold.Load() }

// Available reports whether the GPU NTT path is reachable.
func Available() bool {
	return GPUAvailable()
}

// SubRing -> Montgomery context registry. The ring fast-path consults
// this map to find a per-SubRing GPU context. Construction is the
// caller's responsibility (typically via fhe.NewRingWithGPU or test
// scaffolding).
var (
	subRingRegistryMu sync.RWMutex
	subRingRegistry   = make(map[*ring.SubRing]*MontgomeryNTTContext)
)

// RegisterSubRing binds a Montgomery GPU context to a SubRing. Subsequent
// calls to NTTSubRingForward / NTTSubRingBackward for that SubRing will
// dispatch to the GPU. Returns an existing context if already bound.
func RegisterSubRing(s *ring.SubRing) (*MontgomeryNTTContext, error) {
	subRingRegistryMu.Lock()
	defer subRingRegistryMu.Unlock()
	if existing, ok := subRingRegistry[s]; ok && existing != nil {
		return existing, nil
	}
	ctx, err := NewMontgomeryNTTContext(s)
	if err != nil {
		return nil, err
	}
	subRingRegistry[s] = ctx
	return ctx, nil
}

// UnregisterSubRing removes the binding and closes the underlying GPU
// context.
func UnregisterSubRing(s *ring.SubRing) {
	subRingRegistryMu.Lock()
	defer subRingRegistryMu.Unlock()
	if ctx, ok := subRingRegistry[s]; ok {
		ctx.Close()
		delete(subRingRegistry, s)
	}
}

// LookupSubRing returns the bound GPU context, or nil if none.
func LookupSubRing(s *ring.SubRing) *MontgomeryNTTContext {
	subRingRegistryMu.RLock()
	defer subRingRegistryMu.RUnlock()
	return subRingRegistry[s]
}

// NTTSubRingForward dispatches a forward NTT through the registered
// GPU context for s. Returns false if no context is registered or N is
// below the threshold; the caller should then run the pure-Go path.
func NTTSubRingForward(s *ring.SubRing, src, dst []uint64) bool {
	if !Available() {
		return false
	}
	thr := nttThreshold.Load()
	if thr == 0 || uint32(s.N) < thr {
		return false
	}
	ctx := LookupSubRing(s)
	if ctx == nil {
		return false
	}
	if &src[0] != &dst[0] {
		copy(dst, src)
	}
	if err := ctx.Forward(dst, 1); err != nil {
		return false
	}
	return true
}

// NTTSubRingBackward dispatches an inverse NTT through the registered
// GPU context.
func NTTSubRingBackward(s *ring.SubRing, src, dst []uint64) bool {
	if !Available() {
		return false
	}
	thr := nttThreshold.Load()
	if thr == 0 || uint32(s.N) < thr {
		return false
	}
	ctx := LookupSubRing(s)
	if ctx == nil {
		return false
	}
	if &src[0] != &dst[0] {
		copy(dst, src)
	}
	if err := ctx.Backward(dst, 1); err != nil {
		return false
	}
	return true
}
