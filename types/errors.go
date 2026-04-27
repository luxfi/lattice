// Copyright (c) 2026, Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause

package types

import "errors"

// Sentinel errors for type-system rejections.
//
// These errors are returned by Validate() methods at the boundary between
// caller and kernel dispatch. They MUST be checked with errors.Is, not by
// string comparison.
var (
	// ErrDomainMismatch is returned when a kernel is invoked with input,
	// root, and output domain tags that are not internally consistent.
	ErrDomainMismatch = errors.New("lattice/types: domain mismatch")

	// ErrModulusZero is returned when an NTTContext or ReductionBudget is
	// constructed with modulus == 0.
	ErrModulusZero = errors.New("lattice/types: modulus must be non-zero")

	// ErrNNotPowerOfTwo is returned when an NTTContext is constructed with
	// a polynomial degree N that is not a power of two.
	ErrNNotPowerOfTwo = errors.New("lattice/types: N must be a power of two")

	// ErrModulusTooLargeForLazy is returned when ReductionBudget is asked
	// for a Lazy mode that the modulus does not fit (e.g. Lazy8 requires
	// modulus < 2^29).
	ErrModulusTooLargeForLazy = errors.New("lattice/types: modulus too large for requested lazy mode")
)
