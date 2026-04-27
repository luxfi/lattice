// Copyright (c) 2026, Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause

package types

import "math/bits"

// ReductionMode selects how aggressively a kernel chain may defer modular
// reduction. Lazy reduction is a correctness-sensitive performance
// optimisation: it leaves intermediate values in [0, k*q) for some
// small k > 1, and skips the conditional subtract that would normalise
// each result back to [0, q).
//
// Reference: LP-137-FHE-TYPING.md §5 (lazy reduction modes).
type ReductionMode uint8

const (
	// ReductionStrictEveryOp normalises after every modular operation.
	// Result of each op is in [0, q). Slowest, safest.
	ReductionStrictEveryOp ReductionMode = 0

	// ReductionLazy2 allows result range [0, 2q). 1 deferred subtract
	// allowed before normalisation. Requires q < 2^63.
	ReductionLazy2 ReductionMode = 1

	// ReductionLazy4 allows result range [0, 4q). Up to 2 deferred adds
	// in [0, 2q) chained. Requires q < 2^62.
	ReductionLazy4 ReductionMode = 2

	// ReductionLazy8 allows result range [0, 8q). Up to 3 deferred adds.
	// Requires q < 2^61.
	ReductionLazy8 ReductionMode = 3
)

// String returns the stable name for the reduction mode.
func (m ReductionMode) String() string {
	switch m {
	case ReductionStrictEveryOp:
		return "Strict"
	case ReductionLazy2:
		return "Lazy2"
	case ReductionLazy4:
		return "Lazy4"
	case ReductionLazy8:
		return "Lazy8"
	default:
		return "Unknown"
	}
}

// ReductionBudget tracks how many lazy ops a kernel chain has consumed and
// when it must insert a normalisation step.
//
// Layout is byte-stable across Go and C++.
type ReductionBudget struct {
	// Modulus is q. Used to recompute MaxOpsBeforeOverflow if Mode changes.
	// 8 bytes, offset 0.
	Modulus uint64

	// OpsSinceReduce is the number of lazy ops accumulated since the last
	// explicit reduction.
	// 4 bytes, offset 8.
	OpsSinceReduce uint32

	// MaxOpsBeforeOverflow is the cap derived from (Mode, Modulus) by
	// SafeBoundFor. The kernel must insert a reduction once
	// OpsSinceReduce reaches this value.
	// 4 bytes, offset 12.
	MaxOpsBeforeOverflow uint32

	// Mode is the lazy mode in effect.
	// 1 byte, offset 16.
	Mode ReductionMode

	// _pad pads the struct to 24 bytes (multiple of 8).
	// 7 bytes, offset 17.
	_pad [7]uint8
}

// NewReductionBudget constructs a budget for (mode, modulus). Returns
// ErrModulusZero if modulus == 0, or ErrModulusTooLargeForLazy if the
// modulus exceeds the lazy mode's safe bound.
func NewReductionBudget(mode ReductionMode, modulus uint64) (ReductionBudget, error) {
	if modulus == 0 {
		return ReductionBudget{}, ErrModulusZero
	}
	if !lazyModeFits(mode, modulus) {
		return ReductionBudget{}, ErrModulusTooLargeForLazy
	}
	return ReductionBudget{
		Modulus:              modulus,
		OpsSinceReduce:       0,
		MaxOpsBeforeOverflow: SafeBoundFor(mode, modulus),
		Mode:                 mode,
	}, nil
}

// RemainingOps returns the number of additional lazy ops the kernel may
// perform before NeedsReduce returns true. Saturating at 0.
func (rb *ReductionBudget) RemainingOps() uint32 {
	if rb.OpsSinceReduce >= rb.MaxOpsBeforeOverflow {
		return 0
	}
	return rb.MaxOpsBeforeOverflow - rb.OpsSinceReduce
}

// NeedsReduce reports whether the next op must be preceded by a reduction.
func (rb *ReductionBudget) NeedsReduce() bool {
	return rb.OpsSinceReduce >= rb.MaxOpsBeforeOverflow
}

// Charge increments OpsSinceReduce by n. Caller invokes this each time
// it performs n lazy ops.
func (rb *ReductionBudget) Charge(n uint32) {
	rb.OpsSinceReduce += n
}

// Reset zeroes OpsSinceReduce. Caller invokes this immediately after a
// normalisation pass.
func (rb *ReductionBudget) Reset() {
	rb.OpsSinceReduce = 0
}

// SafeBoundFor returns the maximum number of lazy ops permissible before
// normalisation, given the lazy mode and modulus.
//
// The bounds correspond to the bit budget of a 64-bit limb:
//
//	Strict   : 1            (always reduce)
//	Lazy2    : 1 mult / 2 adds for q < 2^63 (range [0, 2q))
//	Lazy4    : 2 mults     for q < 2^62 (range [0, 4q))
//	Lazy8    : 3 mults     for q < 2^61 (range [0, 8q))
//
// For moduli that do not fit the requested mode (q >= 2^k for the relevant
// k), SafeBoundFor returns 0 to force the caller down the strict path.
func SafeBoundFor(mode ReductionMode, modulus uint64) uint32 {
	if modulus == 0 {
		return 0
	}
	if !lazyModeFits(mode, modulus) {
		return 0
	}
	switch mode {
	case ReductionStrictEveryOp:
		return 1
	case ReductionLazy2:
		return 2
	case ReductionLazy4:
		return 4
	case ReductionLazy8:
		return 8
	default:
		return 0
	}
}

// lazyModeFits reports whether the modulus is small enough for the lazy
// mode without uint64 overflow. The bit thresholds are intentionally
// conservative so that intermediate sums of k operands each in [0, k*q)
// stay strictly below 2^64.
func lazyModeFits(mode ReductionMode, modulus uint64) bool {
	bitlen := uint(bits.Len64(modulus))
	switch mode {
	case ReductionStrictEveryOp:
		return bitlen <= 64
	case ReductionLazy2:
		return bitlen <= 63
	case ReductionLazy4:
		return bitlen <= 62
	case ReductionLazy8:
		return bitlen <= 61
	default:
		return false
	}
}
