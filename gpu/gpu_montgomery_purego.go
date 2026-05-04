//go:build !cgo || !gpu

// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

// Package gpu — pure-Go Montgomery NTT dispatch path.
//
// Default build (no cgo, or no `-tags gpu`) routes Forward/Backward
// through the canonical pure-Go Montgomery NTT in
// github.com/luxfi/lattice/v7/ring (SubRing.NTT / INTT). There is one
// and only one Go implementation of the Montgomery NTT in the Lux
// stack — this file does not re-implement it, only delegates.
//
// Output is byte-equal to the cgo+gpu path (which itself is byte-equal
// to the pure-Go reference by contract). Callers see identical
// semantics regardless of build tags; cgo + gpu is purely an opt-in
// optimization, never a correctness boundary.
package gpu

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/luxfi/lattice/v7/ring"
)

// MontgomeryNTTContext binds a luxfi/lattice/v7/ring.SubRing to the
// pure-Go Montgomery NTT path. Forward and Backward delegate directly
// to ring.SubRing.NTT and ring.SubRing.INTT — the canonical Go
// implementation. There is no re-implementation here; this struct is
// a thin dispatch wrapper that mirrors gpu_montgomery_cgo.go's API so
// callers compile identically across build tags.
type MontgomeryNTTContext struct {
	subring *ring.SubRing
	N       uint32
	Q       uint64
	mu      sync.RWMutex
	closed  atomic.Bool
}

// NewMontgomeryNTTContext binds to a SubRing. The SubRing's existing
// NTT tables are the source of truth — we hold a pointer, never a copy.
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
	// s.N is positive (validated above) and bounded by ring degree
	// (16384 max in production); the int -> uint32 conversion is safe.
	/* #nosec G115 -- s.N is validated above to be > 0, bounded by max ring degree */
	N := uint32(s.N)
	return &MontgomeryNTTContext{
		subring: s,
		N:       N,
		Q:       s.Modulus,
	}, nil
}

// Close releases the context. The pure-Go path holds no external
// resources; Close is idempotent and exists only to mirror the cgo API.
func (ctx *MontgomeryNTTContext) Close() {
	ctx.closed.Store(true)
}

// Forward applies the NTT in-place via ring.SubRing.NTT. data must
// have length batch*N. For batch > 1, processes each polynomial
// sequentially.
func (ctx *MontgomeryNTTContext) Forward(data []uint64, batch uint32) error {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.closed.Load() {
		return fmt.Errorf("context closed")
	}
	N := int(ctx.N)
	if int(batch)*N > len(data) {
		return fmt.Errorf("buffer too small: need %d, got %d",
			int(batch)*N, len(data))
	}
	for i := uint32(0); i < batch; i++ {
		off := int(i) * N
		ctx.subring.NTT(data[off:off+N], data[off:off+N])
	}
	return nil
}

// Backward applies the inverse NTT in-place via ring.SubRing.INTT.
func (ctx *MontgomeryNTTContext) Backward(data []uint64, batch uint32) error {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.closed.Load() {
		return fmt.Errorf("context closed")
	}
	N := int(ctx.N)
	if int(batch)*N > len(data) {
		return fmt.Errorf("buffer too small: need %d, got %d",
			int(batch)*N, len(data))
	}
	for i := uint32(0); i < batch; i++ {
		off := int(i) * N
		ctx.subring.INTT(data[off:off+N], data[off:off+N])
	}
	return nil
}

// =============================================================================
// SubRing dispatch — pure-Go path
// =============================================================================
//
// nttThreshold mirrors the cgo build's signature so callers can probe
// and override consistently across build tags. In the pure-Go build
// the value is honored only by Forward/Backward through an explicit
// MontgomeryNTTContext; the ring core's SubRing.NTT runs on its own
// pure-Go path without consulting this dispatcher.

var nttThreshold atomic.Uint32

// SetNTTThreshold stores the threshold.
func SetNTTThreshold(n uint32) { nttThreshold.Store(n) }

// NTTThreshold returns the current threshold.
func NTTThreshold() uint32 { return nttThreshold.Load() }

// Available reports whether a hardware-accelerated NTT path is
// reachable. False in the pure-Go build: SubRing.NTT is the canonical
// path and the ring core invokes it directly. Returning false here is
// the contract by which NTTSubRingForward / NTTSubRingBackward signal
// "use the default path" to the ring dispatcher.
func Available() bool { return false }

// SubRing -> Montgomery context registry. Mirrors the cgo build's
// surface so callers compile identically across builds.
var (
	subRingRegistryMu sync.RWMutex
	subRingRegistry   = make(map[*ring.SubRing]*MontgomeryNTTContext)
)

// RegisterSubRing binds a Montgomery context to a SubRing. Subsequent
// Forward/Backward calls delegate to the canonical pure-Go NTT in
// ring.SubRing.NTT / INTT.
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

// UnregisterSubRing removes the binding.
func UnregisterSubRing(s *ring.SubRing) {
	subRingRegistryMu.Lock()
	defer subRingRegistryMu.Unlock()
	if ctx, ok := subRingRegistry[s]; ok {
		ctx.Close()
		delete(subRingRegistry, s)
	}
}

// LookupSubRing returns the bound context, or nil if none.
func LookupSubRing(s *ring.SubRing) *MontgomeryNTTContext {
	subRingRegistryMu.RLock()
	defer subRingRegistryMu.RUnlock()
	return subRingRegistry[s]
}

// NTTSubRingForward is the dispatcher hook consulted by ring.SubRing.
// In the pure-Go build it always returns false: SubRing.NTT is the
// canonical pure-Go path and the ring core runs it without our
// involvement. Returning false signals "use the default path."
func NTTSubRingForward(_ *ring.SubRing, _, _ []uint64) bool { return false }

// NTTSubRingBackward is the inverse dispatcher hook. Same contract as
// NTTSubRingForward.
func NTTSubRingBackward(_ *ring.SubRing, _, _ []uint64) bool { return false }
