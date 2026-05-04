// Copyright (c) 2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause

// Package wire is lattice's wire-format hardening boundary.
//
// LP-107 Phase 3 (cycle-free portion): lattice consumes
// luxfi/math/codec for bounded decoding of untrusted wire data.
// luxfi/math/codec does NOT import any lattice package, so this is a
// safe one-way dependency.
//
// The full Phase 3 inversion (lattice/ring delegating to
// luxfi/math/ntt for the canonical NTT body) requires moving ~5K LoC
// of NTT/Montgomery code from lattice/ring into luxfi/math/ntt; that
// migration is queued as a separate work-item, since its surface
// touches every downstream consumer of lattice/ring.
//
// In the meantime, lattice/wire shows that lattice CAN consume the
// math substrate where the dependency direction allows: codec
// hardening lives once, in luxfi/math/codec, and is reused uniformly
// across pulsar, lens, fhe, lattice.
package wire

import (
	"bytes"
	"fmt"

	"github.com/luxfi/math/codec"
)

// MaxLatticeWireUintSliceLen — bound for any uint64 slice that
// crosses the lattice wire boundary. Production NTT degrees max out
// at 16384 (FHE PN13QP54); we cap at 4096 for the consensus-critical
// Pulsar lane and provide a higher-cap helper for FHE.
const MaxLatticeWireUintSliceLen = 4096

// LatticeConsensusWireLimits is the bounded codec.Limits suitable for
// consensus-critical lattice frames (Pulsar Vector[Poly], threshold-
// share blobs).
var LatticeConsensusWireLimits = codec.Limits{
	MaxFrameBytes:     16 * 1024 * 1024,
	MaxUint16SliceLen: MaxLatticeWireUintSliceLen,
	MaxUint32SliceLen: MaxLatticeWireUintSliceLen,
	MaxUint64SliceLen: MaxLatticeWireUintSliceLen,
	MaxDepth:          4,
}

// LatticeFHEWireLimits is the bounded codec.Limits for FHE-class
// lattice frames (RNS chains, EvaluationKey blobs). Larger
// MaxFrameBytes + slice cap reflect the deeper polynomial structure.
var LatticeFHEWireLimits = codec.Limits{
	MaxFrameBytes:     32 * 1024 * 1024,
	MaxUint16SliceLen: 16384,
	MaxUint32SliceLen: 16384,
	MaxUint64SliceLen: 16384,
	MaxDepth:          5,
}

// ValidateUintSliceFrame asserts the outer length prefix of a wire
// frame is within the configured cap; returns an error wrapping
// codec.ErrLimitExceeded when exceeded.
//
// Use LatticeConsensusWireLimits for Pulsar-class data, or
// LatticeFHEWireLimits for FHE-class data.
func ValidateUintSliceFrame(frame []byte, limits codec.Limits) error {
	r, err := codec.NewReader(bytes.NewReader(frame), limits)
	if err != nil {
		return fmt.Errorf("lattice/wire: NewReader: %w", err)
	}
	if _, err := r.ReadUint64Slice(); err != nil {
		return fmt.Errorf("lattice/wire: outer length: %w", err)
	}
	return nil
}
