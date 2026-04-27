//go:build !cgo || !gpu

// Pure-Go stub for the Montgomery GPU dispatch path. Without cgo + gpu,
// the GPU is unavailable; the public API matches gpu_montgomery_cgo.go
// so callers compile identically and the fast-path always reports
// "no GPU".
//
// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
package gpu

import (
	"errors"
	"sync/atomic"

	"github.com/luxfi/lattice/v7/ring"
)

// MontgomeryNTTContext is a no-op type when GPU is unavailable.
type MontgomeryNTTContext struct{}

// NewMontgomeryNTTContext is a no-op stub.
func NewMontgomeryNTTContext(_ *ring.SubRing) (*MontgomeryNTTContext, error) {
	return nil, errors.New("GPU unavailable: build with cgo + -tags gpu")
}

// Close is a no-op.
func (c *MontgomeryNTTContext) Close() {}

// Forward is a no-op stub.
func (c *MontgomeryNTTContext) Forward(_ []uint64, _ uint32) error {
	return errors.New("GPU unavailable")
}

// Backward is a no-op stub.
func (c *MontgomeryNTTContext) Backward(_ []uint64, _ uint32) error {
	return errors.New("GPU unavailable")
}

// nttThreshold is unused in the stub but kept to mirror the cgo path's
// API surface so callers do not need build-tagged code paths.
var nttThreshold atomic.Uint32

// SetNTTThreshold is a no-op when GPU is unavailable.
func SetNTTThreshold(n uint32) { nttThreshold.Store(n) }

// NTTThreshold returns 0 when GPU is unavailable.
func NTTThreshold() uint32 { return 0 }

// Available reports whether the GPU NTT path is reachable. Always false
// in this build.
func Available() bool { return false }

// RegisterSubRing is a no-op stub.
func RegisterSubRing(_ *ring.SubRing) (*MontgomeryNTTContext, error) {
	return nil, errors.New("GPU unavailable")
}

// UnregisterSubRing is a no-op.
func UnregisterSubRing(_ *ring.SubRing) {}

// LookupSubRing always returns nil.
func LookupSubRing(_ *ring.SubRing) *MontgomeryNTTContext { return nil }

// NTTSubRingForward always returns false in the stub.
func NTTSubRingForward(_ *ring.SubRing, _, _ []uint64) bool { return false }

// NTTSubRingBackward always returns false in the stub.
func NTTSubRingBackward(_ *ring.SubRing, _, _ []uint64) bool { return false }
