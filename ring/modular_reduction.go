// Copyright (c) 2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Thin shim over github.com/luxfi/math/ntt/canonical.
// LP-107 Phase 3 inverted the dependency: luxfi/math owns the
// canonical Montgomery / Barrett body. This package re-exports it so
// downstream consumers (luxfi/pulsar, luxfi/fhe, luxfi/threshold,
// luxfi/multiparty, ...) see no source change.

package ring

import (
	"github.com/luxfi/math/ntt/canonical"
)

// MForm switches a to the Montgomery domain by computing a*2^64 mod q.
func MForm(a, q uint64, bredconstant [2]uint64) uint64 {
	return canonical.MForm(a, q, bredconstant)
}

// MFormLazy switches a to the Montgomery domain by computing
// a*2^64 mod q in constant time. Result in [0, 2q-1].
func MFormLazy(a, q uint64, bredconstant [2]uint64) uint64 {
	return canonical.MFormLazy(a, q, bredconstant)
}

// IMForm switches a from the Montgomery domain back to the standard
// domain by computing a*(1/2^64) mod q.
func IMForm(a, q, mredconstant uint64) uint64 {
	return canonical.IMForm(a, q, mredconstant)
}

// IMFormLazy switches a from the Montgomery domain back to the
// standard domain by computing a*(1/2^64) mod q in constant time.
// Result in [0, 2q-1].
func IMFormLazy(a, q, mredconstant uint64) uint64 {
	return canonical.IMFormLazy(a, q, mredconstant)
}

// GenMRedConstant computes the constant (q^-1) mod 2^64 used by MRed.
func GenMRedConstant(q uint64) uint64 {
	return canonical.GenMRedConstant(q)
}

// MRed computes x * y * (1/2^64) mod q.
func MRed(x, y, q, mredconstant uint64) uint64 {
	return canonical.MRed(x, y, q, mredconstant)
}

// MRedLazy computes x * y * (1/2^64) mod q in constant time.
// Result in [0, 2q-1].
func MRedLazy(x, y, q, mredconstant uint64) uint64 {
	return canonical.MRedLazy(x, y, q, mredconstant)
}

// GenBRedConstant computes the constant pair for the BRed algorithm.
// Returns ((2^128)/q)/(2^64) and (2^128)/q mod 2^64.
func GenBRedConstant(q uint64) [2]uint64 {
	return canonical.GenBRedConstant(q)
}

// BRedAdd computes a mod q.
func BRedAdd(a, q uint64, bredconstant [2]uint64) uint64 {
	return canonical.BRedAdd(a, q, bredconstant)
}

// BRedAddLazy computes a mod q in constant time. Result in [0, 2q-1].
func BRedAddLazy(x, q uint64, bredconstant [2]uint64) uint64 {
	return canonical.BRedAddLazy(x, q, bredconstant)
}

// BRed computes x*y mod q.
func BRed(x, y, q uint64, bredconstant [2]uint64) uint64 {
	return canonical.BRed(x, y, q, bredconstant)
}

// BRedLazy computes x*y mod q in constant time. Result in [0, 2q-1].
func BRedLazy(x, y, q uint64, bredconstant [2]uint64) uint64 {
	return canonical.BRedLazy(x, y, q, bredconstant)
}

// CRed reduces a in [0, 2q-1] to [0, q-1].
func CRed(a, q uint64) uint64 {
	return canonical.CRed(a, q)
}
