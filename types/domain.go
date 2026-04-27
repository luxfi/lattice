// Package types defines the FHE-GPU domain typing system that prevents
// silent correctness failures in NTT/Montgomery dispatch.
//
// Reference: LP-137-FHE-TYPING.md.
//
// Core invariant: every polynomial buffer carries an explicit PolyDomain tag,
// and every NTT/arithmetic kernel declares the (Input, Root, Output) tuple
// it accepts. Dispatch with mismatched tags is rejected at the type-system
// boundary, not silently corrupted at the math layer.
//
// Copyright (c) 2026, Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
package types

// PolyDomain identifies which arithmetic domain a polynomial buffer lives in.
//
// The four domains form a 2x2 grid:
//
//	                  | coefficient form | NTT (evaluation) form
//	------------------+------------------+----------------------
//	standard form     | Standard         | NTTStandard
//	Montgomery form   | Montgomery       | NTTMontgomery
//
// "standard form" means values in [0, q). "Montgomery form" means values
// represented as a*R mod q, where R = 2^64 mod q, used by Montgomery reduction
// (MRed). NTT/non-NTT denotes whether the buffer is in coefficient form or
// in pointwise (evaluation) form after a forward NTT.
//
// Domain mismatches are correctness bugs, not performance bugs. A polynomial
// in Montgomery form cannot be multiplied pointwise with one in standard form;
// the result is silently wrong.
type PolyDomain uint8

const (
	// PolyDomainStandard: coefficient-form, values in [0, q).
	PolyDomainStandard PolyDomain = 0

	// PolyDomainMontgomery: coefficient-form, Montgomery-encoded (a*R mod q).
	PolyDomainMontgomery PolyDomain = 1

	// PolyDomainNTTStandard: NTT/evaluation-form, values in [0, q).
	PolyDomainNTTStandard PolyDomain = 2

	// PolyDomainNTTMontgomery: NTT/evaluation-form, Montgomery-encoded.
	// This is the form produced by Ring.NTT in luxfi/lattice and consumed by
	// Montgomery-domain pointwise multiplication.
	PolyDomainNTTMontgomery PolyDomain = 3
)

// String returns a stable, human-readable name for the domain.
// The string values are part of the wire/log contract; do not rename.
func (d PolyDomain) String() string {
	switch d {
	case PolyDomainStandard:
		return "Standard"
	case PolyDomainMontgomery:
		return "Montgomery"
	case PolyDomainNTTStandard:
		return "NTTStandard"
	case PolyDomainNTTMontgomery:
		return "NTTMontgomery"
	default:
		return "Unknown"
	}
}

// IsNTT reports whether the domain is in NTT/evaluation form.
func (d PolyDomain) IsNTT() bool {
	return d == PolyDomainNTTStandard || d == PolyDomainNTTMontgomery
}

// IsMontgomery reports whether the domain uses Montgomery encoding.
func (d PolyDomain) IsMontgomery() bool {
	return d == PolyDomainMontgomery || d == PolyDomainNTTMontgomery
}

// AfterForwardNTT returns the domain a Standard or Montgomery buffer enters
// after a forward NTT. It is the type-system inverse of AfterInverseNTT.
//
// Standard       -> NTTStandard
// Montgomery     -> NTTMontgomery
// NTT*           -> PolyDomain(0xFF)  // already NTT, caller error
func (d PolyDomain) AfterForwardNTT() PolyDomain {
	switch d {
	case PolyDomainStandard:
		return PolyDomainNTTStandard
	case PolyDomainMontgomery:
		return PolyDomainNTTMontgomery
	default:
		return PolyDomain(0xFF)
	}
}

// AfterInverseNTT returns the domain a NTTStandard or NTTMontgomery buffer
// enters after an inverse NTT.
//
// NTTStandard    -> Standard
// NTTMontgomery  -> Montgomery
// non-NTT        -> PolyDomain(0xFF)  // not in NTT form, caller error
func (d PolyDomain) AfterInverseNTT() PolyDomain {
	switch d {
	case PolyDomainNTTStandard:
		return PolyDomainStandard
	case PolyDomainNTTMontgomery:
		return PolyDomainMontgomery
	default:
		return PolyDomain(0xFF)
	}
}
