// Copyright (c) 2026, Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause

package types

import (
	"bytes"
	"testing"
	"unsafe"
)

// TestNTTContext_CrossLangByteImage asserts that the byte image of a
// canonical NTTContext is identical between Go and C++.
//
// The matching C++ test is at luxcpp/lattice/test/types_test.cpp
// (test_ntt_context_memcmp). If either side changes the layout, both this
// Go test and the C++ memcmp test MUST be updated together.
func TestNTTContext_CrossLangByteImage(t *testing.T) {
	// Same field values as the C++ test_ntt_context_memcmp case.
	ctx := NTTContext{
		Modulus:       0x0123456789ABCDEF,
		MontR:         0x1111111111111111,
		MontR2:        0x2222222222222222,
		QInv:          0x3333333333333333,
		N:             0x00010000,
		ModulusID:     0x00000007,
		TwiddleOffset: 0x000000AB,
		InputDomain:   PolyDomainMontgomery,
		RootDomain:    PolyDomainMontgomery,
		OutputDomain:  PolyDomainNTTMontgomery,
	}

	// Expected little-endian byte image, byte-identical to the C++ side's
	// `want` array.
	want := []byte{
		// modulus 0x0123456789ABCDEF
		0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01,
		// mont_R 0x1111111111111111
		0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
		// mont_R2 0x2222222222222222
		0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22,
		// q_inv 0x3333333333333333
		0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33,
		// N 0x00010000
		0x00, 0x00, 0x01, 0x00,
		// modulus_id 0x00000007
		0x07, 0x00, 0x00, 0x00,
		// twiddle_offset 0x000000AB
		0xAB, 0x00, 0x00, 0x00,
		// input_domain, root_domain, output_domain, _pad
		0x01, 0x01, 0x03, 0x00,
	}

	// Reinterpret ctx as raw bytes via unsafe.Pointer.
	got := unsafe.Slice((*byte)(unsafe.Pointer(&ctx)), unsafe.Sizeof(ctx))
	if !bytes.Equal(got, want) {
		t.Fatalf("NTTContext byte image mismatch with C++ side\n  got:  %x\n  want: %x", got, want)
	}
}
