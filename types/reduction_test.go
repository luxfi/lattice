// Copyright (c) 2026, Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause

package types

import (
	"errors"
	"testing"
	"unsafe"
)

func TestReductionMode_String(t *testing.T) {
	cases := []struct {
		m    ReductionMode
		want string
	}{
		{ReductionStrictEveryOp, "Strict"},
		{ReductionLazy2, "Lazy2"},
		{ReductionLazy4, "Lazy4"},
		{ReductionLazy8, "Lazy8"},
		{ReductionMode(0xFF), "Unknown"},
	}
	for _, c := range cases {
		if got := c.m.String(); got != c.want {
			t.Errorf("ReductionMode(%d) = %q, want %q", c.m, got, c.want)
		}
	}
}

func TestSafeBoundFor_StrictAlwaysOne(t *testing.T) {
	moduli := []uint64{
		0x10001,            // 17 bits
		0x7FFFFFFFFFFFFFC5, // 63 bits, prime
		0xFFFFFFFFFFFFFFC5, // 64 bits, prime
	}
	for _, q := range moduli {
		if got := SafeBoundFor(ReductionStrictEveryOp, q); got != 1 {
			t.Errorf("SafeBoundFor(Strict, q=0x%x) = %d, want 1", q, got)
		}
	}
}

func TestSafeBoundFor_LazyByModulusBitlen(t *testing.T) {
	cases := []struct {
		mode    ReductionMode
		modulus uint64
		want    uint32
	}{
		// 30-bit modulus fits all lazy modes
		{ReductionLazy2, (1 << 30) - 35, 2},
		{ReductionLazy4, (1 << 30) - 35, 4},
		{ReductionLazy8, (1 << 30) - 35, 8},

		// 61-bit modulus is the upper bound for Lazy8
		{ReductionLazy8, (1 << 61) - 1, 8},

		// 62-bit modulus exceeds Lazy8 budget
		{ReductionLazy8, (1 << 62) - 1, 0},

		// 62-bit fits Lazy4
		{ReductionLazy4, (1 << 62) - 1, 4},

		// 63-bit exceeds Lazy4
		{ReductionLazy4, (1 << 63) - 1, 0},

		// 63-bit fits Lazy2
		{ReductionLazy2, (1 << 63) - 1, 2},

		// 64-bit exceeds Lazy2
		{ReductionLazy2, ^uint64(0), 0},

		// modulus 0 always returns 0
		{ReductionLazy4, 0, 0},
	}
	for _, c := range cases {
		if got := SafeBoundFor(c.mode, c.modulus); got != c.want {
			t.Errorf("SafeBoundFor(%s, q=0x%x) = %d, want %d",
				c.mode, c.modulus, got, c.want)
		}
	}
}

func TestNewReductionBudget_RejectsZeroModulus(t *testing.T) {
	if _, err := NewReductionBudget(ReductionLazy4, 0); !errors.Is(err, ErrModulusZero) {
		t.Errorf("expected ErrModulusZero, got %v", err)
	}
}

func TestNewReductionBudget_RejectsTooLargeModulus(t *testing.T) {
	// 64-bit modulus, request Lazy2 (caps at 63-bit) => reject.
	_, err := NewReductionBudget(ReductionLazy2, ^uint64(0))
	if !errors.Is(err, ErrModulusTooLargeForLazy) {
		t.Errorf("expected ErrModulusTooLargeForLazy, got %v", err)
	}
}

func TestReductionBudget_ChargeAndReset(t *testing.T) {
	rb, err := NewReductionBudget(ReductionLazy4, (1 << 30) - 35)
	if err != nil {
		t.Fatalf("NewReductionBudget: %v", err)
	}
	if rb.NeedsReduce() {
		t.Errorf("fresh budget should not require reduction")
	}
	if got := rb.RemainingOps(); got != 4 {
		t.Errorf("RemainingOps = %d, want 4", got)
	}
	rb.Charge(2)
	if got := rb.RemainingOps(); got != 2 {
		t.Errorf("after charge 2, RemainingOps = %d, want 2", got)
	}
	rb.Charge(2)
	if !rb.NeedsReduce() {
		t.Errorf("after charge 4 with cap 4, should need reduce")
	}
	if got := rb.RemainingOps(); got != 0 {
		t.Errorf("RemainingOps saturated should be 0, got %d", got)
	}
	rb.Reset()
	if rb.NeedsReduce() {
		t.Errorf("after reset, should not need reduce")
	}
	if got := rb.RemainingOps(); got != 4 {
		t.Errorf("after reset, RemainingOps = %d, want 4", got)
	}
}

func TestReductionBudget_RemainingOpsSaturates(t *testing.T) {
	rb, err := NewReductionBudget(ReductionLazy2, 1<<30)
	if err != nil {
		t.Fatalf("NewReductionBudget: %v", err)
	}
	rb.Charge(1000) // way over cap
	if got := rb.RemainingOps(); got != 0 {
		t.Errorf("over-charged budget RemainingOps = %d, want 0", got)
	}
	if !rb.NeedsReduce() {
		t.Errorf("over-charged budget should need reduce")
	}
}

// TestReductionBudget_Layout pins layout for cgo compatibility.
func TestReductionBudget_Layout(t *testing.T) {
	const wantSize = 24
	if got := unsafe.Sizeof(ReductionBudget{}); got != wantSize {
		t.Fatalf("sizeof(ReductionBudget) = %d, want %d", got, wantSize)
	}
	var rb ReductionBudget
	base := uintptr(unsafe.Pointer(&rb))
	cases := []struct {
		name string
		off  uintptr
		want uintptr
	}{
		{"Modulus", uintptr(unsafe.Pointer(&rb.Modulus)) - base, 0},
		{"OpsSinceReduce", uintptr(unsafe.Pointer(&rb.OpsSinceReduce)) - base, 8},
		{"MaxOpsBeforeOverflow", uintptr(unsafe.Pointer(&rb.MaxOpsBeforeOverflow)) - base, 12},
		{"Mode", uintptr(unsafe.Pointer(&rb.Mode)) - base, 16},
	}
	for _, c := range cases {
		if c.off != c.want {
			t.Errorf("offset of %s = %d, want %d", c.name, c.off, c.want)
		}
	}
}
