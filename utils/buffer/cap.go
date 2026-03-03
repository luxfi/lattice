// Copyright (c) Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

package buffer

import (
	"errors"
	"os"
	"strconv"
	"sync/atomic"
)

// DefaultMaxSliceLen is the default maximum slice length accepted by the
// length-bounded slice readers (Read{Uint16,Uint32,Uint64}SliceBounded). It
// is intentionally large enough to cover all in-tree lattice serialization
// paths (a Poly with N=2^17 coefficients only needs 131072 entries) while
// still rejecting attacker-controlled length prefixes that would force the
// callee to allocate gigabytes or recurse without bound.
//
// The cap is a per-call ceiling on the destination slice length, NOT the
// total bytes peeked from the reader. It exists to give downstream callers
// (e.g. luxfi/warp) a predictable upper bound when they delegate parsing to
// this package without the full lattigo Poly wire walker.
const DefaultMaxSliceLen = 1 << 20 // 1,048,576 entries

// EnvMaxSliceLen overrides DefaultMaxSliceLen at process start when set to
// a positive integer. Values <= 0 fall back to the default. Reading the
// environment is done once per process; subsequent changes are ignored.
const EnvMaxSliceLen = "LUX_LATTICE_MAX_UINT64_SLICE_LEN"

var (
	// ErrSliceTooLarge is returned by the length-bounded slice readers when
	// the requested destination length exceeds the configured cap.
	ErrSliceTooLarge = errors.New("buffer: slice length exceeds maximum")

	// ErrZeroProgress is returned by ReadUint{16,32,64}Slice when the
	// underlying Reader returns a partial Peek that cannot decode at least
	// one element. Without this guard, the slice readers would recurse on
	// the unchanged tail (c[buffered:] with buffered == 0) and exhaust the
	// goroutine stack on an attacker-controlled wire prefix.
	//
	// Discovered via fuzzing in github.com/luxfi/warp/pulsar
	// (FuzzPulseDeserialize, regression seed ccdb090e0ca0007b, 2026-03-03).
	ErrZeroProgress = errors.New("buffer: zero-progress slice read (short input)")

	maxSliceLen atomic.Int64
)

func init() {
	maxSliceLen.Store(int64(DefaultMaxSliceLen))
	if v := os.Getenv(EnvMaxSliceLen); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			maxSliceLen.Store(n)
		}
	}
}

// MaxSliceLen returns the currently configured per-call slice cap.
func MaxSliceLen() int { return int(maxSliceLen.Load()) }

// SetMaxSliceLen overrides the configured cap. n <= 0 resets to default.
// Intended for tests and for callers that need a tighter bound than the
// default 1 MiB.
func SetMaxSliceLen(n int) {
	if n <= 0 {
		maxSliceLen.Store(int64(DefaultMaxSliceLen))
		return
	}
	maxSliceLen.Store(int64(n))
}
