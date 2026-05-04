// Copyright (c) 2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause

package wire

import (
	"bytes"
	"errors"
	"testing"

	"github.com/luxfi/math/codec"
)

func encodeUvarint(out *bytes.Buffer, v uint64) {
	for v >= 0x80 {
		out.WriteByte(byte(v) | 0x80)
		v >>= 7
	}
	out.WriteByte(byte(v))
}

// TestValidateUintSliceFrame_RejectsHugeLength is the regression test
// for lattice issue #4 (Vector[T].ReadFrom unbounded allocation),
// now centralized through luxfi/math/codec.Reader. The lattice repo
// itself can now reject the same attack input class through the
// substrate — completing the LP-107 cycle: pulsar, lens, fhe, AND
// lattice all reject the same byte input identically.
func TestValidateUintSliceFrame_RejectsHugeLength(t *testing.T) {
	const huge = uint64(70_368_955_777_453)
	var buf bytes.Buffer
	encodeUvarint(&buf, huge)

	err := ValidateUintSliceFrame(buf.Bytes(), LatticeConsensusWireLimits)
	if err == nil {
		t.Fatal("ValidateUintSliceFrame returned nil for huge length")
	}
	if !errors.Is(err, codec.ErrLimitExceeded) {
		t.Errorf("err is not ErrLimitExceeded: %v", err)
	}
}

func TestValidateUintSliceFrame_FHELimits_AcceptsLarger(t *testing.T) {
	// FHE limits accept up to 16384; consensus limits cap at 4096.
	// A frame requesting 8192 should pass FHE limits, fail consensus.
	var buf bytes.Buffer
	encodeUvarint(&buf, 8192)
	// Provide payload bytes (not strictly required if reading errors
	// before exhausting the buffer, but completeness).
	buf.Write(make([]byte, 8192*8))
	if err := ValidateUintSliceFrame(buf.Bytes(), LatticeFHEWireLimits); err != nil {
		t.Errorf("FHE limits should accept 8192: %v", err)
	}
	// Same bytes via consensus limits → should reject.
	if err := ValidateUintSliceFrame(buf.Bytes(), LatticeConsensusWireLimits); err == nil {
		t.Error("consensus limits should reject 8192")
	}
}
