// Copyright (c) 2026, Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause

package types

import (
	"errors"
	"testing"
	"unsafe"
)

// TestPolyDomain_StableValues asserts the on-the-wire byte values for each
// PolyDomain variant. Changing these breaks ciphertext header digests and
// kernel-tag dispatch tables. DO NOT change.
func TestPolyDomain_StableValues(t *testing.T) {
	cases := []struct {
		name string
		got  PolyDomain
		want uint8
	}{
		{"Standard", PolyDomainStandard, 0},
		{"Montgomery", PolyDomainMontgomery, 1},
		{"NTTStandard", PolyDomainNTTStandard, 2},
		{"NTTMontgomery", PolyDomainNTTMontgomery, 3},
	}
	for _, c := range cases {
		if uint8(c.got) != c.want {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
		}
	}
}

// TestPolyDomain_StableStrings asserts the human-readable name returned by
// String(). These strings are part of the log/wire contract.
func TestPolyDomain_StableStrings(t *testing.T) {
	cases := []struct {
		d    PolyDomain
		want string
	}{
		{PolyDomainStandard, "Standard"},
		{PolyDomainMontgomery, "Montgomery"},
		{PolyDomainNTTStandard, "NTTStandard"},
		{PolyDomainNTTMontgomery, "NTTMontgomery"},
		{PolyDomain(0xFF), "Unknown"},
	}
	for _, c := range cases {
		if got := c.d.String(); got != c.want {
			t.Errorf("PolyDomain(%d).String() = %q, want %q", c.d, got, c.want)
		}
	}
}

func TestPolyDomain_NTTAndMontgomeryFlags(t *testing.T) {
	cases := []struct {
		d           PolyDomain
		wantNTT     bool
		wantMontgom bool
	}{
		{PolyDomainStandard, false, false},
		{PolyDomainMontgomery, false, true},
		{PolyDomainNTTStandard, true, false},
		{PolyDomainNTTMontgomery, true, true},
	}
	for _, c := range cases {
		if got := c.d.IsNTT(); got != c.wantNTT {
			t.Errorf("%s.IsNTT() = %v, want %v", c.d, got, c.wantNTT)
		}
		if got := c.d.IsMontgomery(); got != c.wantMontgom {
			t.Errorf("%s.IsMontgomery() = %v, want %v", c.d, got, c.wantMontgom)
		}
	}
}

func TestPolyDomain_ForwardInverseRoundTrip(t *testing.T) {
	cases := []PolyDomain{PolyDomainStandard, PolyDomainMontgomery}
	for _, in := range cases {
		ntt := in.AfterForwardNTT()
		if ntt == PolyDomain(0xFF) {
			t.Fatalf("AfterForwardNTT(%s) returned invalid", in)
		}
		back := ntt.AfterInverseNTT()
		if back != in {
			t.Errorf("round trip %s -> %s -> %s", in, ntt, back)
		}
	}
}

func TestPolyDomain_ForwardOnAlreadyNTT(t *testing.T) {
	if got := PolyDomainNTTStandard.AfterForwardNTT(); got != PolyDomain(0xFF) {
		t.Errorf("forward on NTTStandard should be invalid, got %s", got)
	}
	if got := PolyDomainNTTMontgomery.AfterForwardNTT(); got != PolyDomain(0xFF) {
		t.Errorf("forward on NTTMontgomery should be invalid, got %s", got)
	}
}

func TestNTTContext_ValidatesModulusZero(t *testing.T) {
	ctx := &NTTContext{N: 16}
	if err := ctx.Validate(); !errors.Is(err, ErrModulusZero) {
		t.Errorf("modulus=0 -> err=%v, want ErrModulusZero", err)
	}
}

func TestNTTContext_ValidatesNPowerOfTwo(t *testing.T) {
	bad := []uint32{0, 1, 3, 5, 7, 100, 1000}
	for _, n := range bad {
		ctx := &NTTContext{Modulus: 0xFFFFFFFFFFFFFFC5, N: n}
		if err := ctx.Validate(); !errors.Is(err, ErrNNotPowerOfTwo) {
			t.Errorf("N=%d -> err=%v, want ErrNNotPowerOfTwo", n, err)
		}
	}
}

func TestNTTContext_ValidatesRootDomainMatchesInput(t *testing.T) {
	ctx := &NTTContext{
		Modulus:      0xFFFFFFFFFFFFFFC5,
		N:            16,
		InputDomain:  PolyDomainStandard,
		RootDomain:   PolyDomainMontgomery, // mismatch
		OutputDomain: PolyDomainNTTStandard,
	}
	if err := ctx.Validate(); !errors.Is(err, ErrDomainMismatch) {
		t.Errorf("root/input mismatch -> err=%v, want ErrDomainMismatch", err)
	}
}

func TestNTTContext_ValidatesOutputDomainConsistency(t *testing.T) {
	cases := []struct {
		name   string
		input  PolyDomain
		output PolyDomain
		ok     bool
	}{
		{"std fwd", PolyDomainStandard, PolyDomainNTTStandard, true},
		{"std fwd wrong out", PolyDomainStandard, PolyDomainNTTMontgomery, false},
		{"mont fwd", PolyDomainMontgomery, PolyDomainNTTMontgomery, true},
		{"mont fwd wrong out", PolyDomainMontgomery, PolyDomainNTTStandard, false},
		{"std inv", PolyDomainNTTStandard, PolyDomainStandard, true},
		{"std inv wrong out", PolyDomainNTTStandard, PolyDomainMontgomery, false},
		{"mont inv", PolyDomainNTTMontgomery, PolyDomainMontgomery, true},
		{"mont inv wrong out", PolyDomainNTTMontgomery, PolyDomainStandard, false},
	}
	for _, c := range cases {
		ctx := &NTTContext{
			Modulus:      0xFFFFFFFFFFFFFFC5,
			N:            16,
			InputDomain:  c.input,
			RootDomain:   c.input,
			OutputDomain: c.output,
		}
		err := ctx.Validate()
		if c.ok && err != nil {
			t.Errorf("%s: expected ok, got %v", c.name, err)
		}
		if !c.ok && !errors.Is(err, ErrDomainMismatch) {
			t.Errorf("%s: expected ErrDomainMismatch, got %v", c.name, err)
		}
	}
}

func TestNTTContext_ForwardInverseClassification(t *testing.T) {
	fwd := &NTTContext{
		Modulus: 0xFFFFFFFFFFFFFFC5, N: 16,
		InputDomain: PolyDomainMontgomery, RootDomain: PolyDomainMontgomery,
		OutputDomain: PolyDomainNTTMontgomery,
	}
	if !fwd.IsForward() || fwd.IsInverse() {
		t.Errorf("expected forward, got fwd=%v inv=%v", fwd.IsForward(), fwd.IsInverse())
	}
	inv := &NTTContext{
		Modulus: 0xFFFFFFFFFFFFFFC5, N: 16,
		InputDomain: PolyDomainNTTMontgomery, RootDomain: PolyDomainNTTMontgomery,
		OutputDomain: PolyDomainMontgomery,
	}
	if inv.IsForward() || !inv.IsInverse() {
		t.Errorf("expected inverse, got fwd=%v inv=%v", inv.IsForward(), inv.IsInverse())
	}
}

// TestNTTContext_Layout pins the byte layout of NTTContext. Failing this
// test means the Go and C++ sides will silently disagree on field
// offsets via cgo. STOP and fix the layout, do not paper over.
func TestNTTContext_Layout(t *testing.T) {
	const wantSize = 48
	if got := unsafe.Sizeof(NTTContext{}); got != wantSize {
		t.Fatalf("sizeof(NTTContext) = %d, want %d", got, wantSize)
	}
	var ctx NTTContext
	base := uintptr(unsafe.Pointer(&ctx))
	cases := []struct {
		name string
		off  uintptr
		want uintptr
	}{
		{"Modulus", uintptr(unsafe.Pointer(&ctx.Modulus)) - base, 0},
		{"MontR", uintptr(unsafe.Pointer(&ctx.MontR)) - base, 8},
		{"MontR2", uintptr(unsafe.Pointer(&ctx.MontR2)) - base, 16},
		{"QInv", uintptr(unsafe.Pointer(&ctx.QInv)) - base, 24},
		{"N", uintptr(unsafe.Pointer(&ctx.N)) - base, 32},
		{"ModulusID", uintptr(unsafe.Pointer(&ctx.ModulusID)) - base, 36},
		{"TwiddleOffset", uintptr(unsafe.Pointer(&ctx.TwiddleOffset)) - base, 40},
		{"InputDomain", uintptr(unsafe.Pointer(&ctx.InputDomain)) - base, 44},
		{"RootDomain", uintptr(unsafe.Pointer(&ctx.RootDomain)) - base, 45},
		{"OutputDomain", uintptr(unsafe.Pointer(&ctx.OutputDomain)) - base, 46},
	}
	for _, c := range cases {
		if c.off != c.want {
			t.Errorf("offset of %s = %d, want %d", c.name, c.off, c.want)
		}
	}
}
