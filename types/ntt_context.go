// Copyright (c) 2026, Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause

package types

import "math/bits"

// NTTContext is the typed handle that every NTT/arithmetic kernel must accept
// in place of an opaque pointer + a pile of scalars.
//
// Layout is byte-stable across Go and C++ for cgo / shared-memory dispatch.
// See lux/lattice/types/domain_layout_test.go and
// luxcpp/lattice/include/lux/lattice/types/domain.hpp for the matching
// static_assert. The struct has no padding (all fields fall on natural
// alignment for an 8-byte aligned struct).
//
// Field ordering MUST NOT change without updating the C++ mirror and the
// layout tests on both sides.
type NTTContext struct {
	// Modulus is the prime q used by this NTT context.
	// 8 bytes, offset 0.
	Modulus uint64

	// MontR is R mod q where R = 2^64.
	// 8 bytes, offset 8.
	MontR uint64

	// MontR2 is R^2 mod q. Used to convert to/from Montgomery form
	// without a 128-bit divide.
	// 8 bytes, offset 16.
	MontR2 uint64

	// QInv is q^{-1} mod 2^64 (the constant required for Montgomery
	// reduction MRed). The lattice/v7 ring package calls this
	// "mredconstant".
	// 8 bytes, offset 24.
	QInv uint64

	// N is the polynomial ring degree (must be a power of two).
	// 4 bytes, offset 32.
	N uint32

	// ModulusID is an opaque identifier into a modulus chain (RNS basis).
	// Kernels use it to look up twiddle tables, BRedConstants, etc.
	// 4 bytes, offset 36.
	ModulusID uint32

	// TwiddleOffset is the offset into a flat twiddle-factor table where
	// this context's roots begin. Allows multiple contexts to share a
	// single contiguous twiddle buffer on GPU.
	// 4 bytes, offset 40.
	TwiddleOffset uint32

	// InputDomain is the domain the kernel expects on its input buffer.
	// 1 byte, offset 44.
	InputDomain PolyDomain

	// RootDomain is the domain the twiddle (root) table lives in.
	// 1 byte, offset 45.
	RootDomain PolyDomain

	// OutputDomain is the domain the kernel will leave the output buffer
	// in after dispatch.
	// 1 byte, offset 46.
	OutputDomain PolyDomain

	// _pad ensures total size is 48 bytes (multiple of 8).
	// 1 byte, offset 47.
	_pad uint8
}

// Validate enforces type-system invariants at the kernel boundary.
//
// A valid NTTContext satisfies:
//  1. Modulus != 0
//  2. N is a power of two and N >= 2
//  3. RootDomain matches InputDomain (twiddles and operand share encoding)
//  4. OutputDomain is consistent with the input/root pair under the
//     forward-NTT transition rule (Montgomery in -> NTTMontgomery out;
//     Standard in -> NTTStandard out; NTT in -> coefficient out)
//
// Returns nil on success, or an ErrDomainMismatch / ErrModulusZero /
// ErrNNotPowerOfTwo error otherwise. Callers MUST check this before
// dispatching to a GPU kernel.
func (ctx *NTTContext) Validate() error {
	if ctx.Modulus == 0 {
		return ErrModulusZero
	}
	if ctx.N < 2 || bits.OnesCount32(ctx.N) != 1 {
		return ErrNNotPowerOfTwo
	}

	// Twiddles must live in the same encoding as the input. A standard-form
	// input multiplied against Montgomery-form roots produces silent
	// corruption; this is the #121 class of bug.
	if ctx.RootDomain != ctx.InputDomain {
		return ErrDomainMismatch
	}

	// Output must be consistent with the direction implied by Input.
	switch ctx.InputDomain {
	case PolyDomainStandard:
		if ctx.OutputDomain != PolyDomainNTTStandard {
			return ErrDomainMismatch
		}
	case PolyDomainMontgomery:
		if ctx.OutputDomain != PolyDomainNTTMontgomery {
			return ErrDomainMismatch
		}
	case PolyDomainNTTStandard:
		if ctx.OutputDomain != PolyDomainStandard {
			return ErrDomainMismatch
		}
	case PolyDomainNTTMontgomery:
		if ctx.OutputDomain != PolyDomainMontgomery {
			return ErrDomainMismatch
		}
	default:
		return ErrDomainMismatch
	}

	return nil
}

// IsForward reports whether this context describes a forward NTT
// (coefficient -> evaluation form).
func (ctx *NTTContext) IsForward() bool {
	return !ctx.InputDomain.IsNTT() && ctx.OutputDomain.IsNTT()
}

// IsInverse reports whether this context describes an inverse NTT
// (evaluation -> coefficient form).
func (ctx *NTTContext) IsInverse() bool {
	return ctx.InputDomain.IsNTT() && !ctx.OutputDomain.IsNTT()
}
