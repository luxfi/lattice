// Copyright (c) 2026, Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause

package types

import (
	"bytes"
	"testing"
	"unsafe"
)

// TestComputeTrustMode_StableValues pins the underlying integer values of
// every ComputeTrustMode variant. The C++ mirror at
// luxcpp/lattice/include/lux/lattice/types/trust.hpp asserts the same
// values; if either side drifts, dispatch silently misclassifies workers
// at the trust boundary.
func TestComputeTrustMode_StableValues(t *testing.T) {
	cases := []struct {
		mode ComputeTrustMode
		want uint8
	}{
		{TrustPublicDeterministic, 0},
		{TrustAttestedGpuOnly, 1},
		{TrustCpuGpuCompositeTEE, 2},
		{TrustConfidentialIO, 3},
		{TrustZKOrFraudProofed, 4},
	}
	for _, c := range cases {
		if uint8(c.mode) != c.want {
			t.Errorf("%s: got %d want %d", c.mode, uint8(c.mode), c.want)
		}
	}
}

// TestComputeTrustMode_Order verifies that trust modes form a strict
// monotonic chain. Eligible() in the registry uses `>=` on this ordering;
// if two variants compare equal, the gate degrades to a substring match.
func TestComputeTrustMode_Order(t *testing.T) {
	chain := []ComputeTrustMode{
		TrustPublicDeterministic,
		TrustAttestedGpuOnly,
		TrustCpuGpuCompositeTEE,
		TrustConfidentialIO,
		TrustZKOrFraudProofed,
	}
	for i := 1; i < len(chain); i++ {
		if !(chain[i] > chain[i-1]) {
			t.Fatalf("trust ordering broken at %s -> %s", chain[i-1], chain[i])
		}
	}
}

func TestComputeTrustMode_String(t *testing.T) {
	cases := []struct {
		mode ComputeTrustMode
		want string
	}{
		{TrustPublicDeterministic, "PublicDeterministic"},
		{TrustAttestedGpuOnly, "AttestedGpuOnly"},
		{TrustCpuGpuCompositeTEE, "CpuGpuCompositeTEE"},
		{TrustConfidentialIO, "ConfidentialIO"},
		{TrustZKOrFraudProofed, "ZKOrFraudProofed"},
		{ComputeTrustMode(99), "Unknown"},
	}
	for _, c := range cases {
		if got := c.mode.String(); got != c.want {
			t.Errorf("%d: got %q want %q", uint8(c.mode), got, c.want)
		}
	}
}

func TestComputeTrustMode_Valid(t *testing.T) {
	for m := ComputeTrustMode(0); m <= TrustZKOrFraudProofed; m++ {
		if !m.Valid() {
			t.Errorf("expected %s to be valid", m)
		}
	}
	if ComputeTrustMode(99).Valid() {
		t.Error("expected 99 to be invalid")
	}
}

// TestConfidentialIOLevel_StableValues pins the underlying integer values
// of every ConfidentialIOLevel variant. Mirrors the C++ static_assert on
// the same enum.
func TestConfidentialIOLevel_StableValues(t *testing.T) {
	cases := []struct {
		level ConfidentialIOLevel
		want  uint8
	}{
		{IOLevelNone, 0},
		{IOLevelCpuTeeOnly, 1},
		{IOLevelCpuGpuComposite, 2},
		{IOLevelProtectedCpuGpuTransfer, 3},
		{IOLevelFullDeviceIOAttested, 4},
	}
	for _, c := range cases {
		if uint8(c.level) != c.want {
			t.Errorf("%s: got %d want %d", c.level, uint8(c.level), c.want)
		}
	}
}

func TestConfidentialIOLevel_Order(t *testing.T) {
	chain := []ConfidentialIOLevel{
		IOLevelNone,
		IOLevelCpuTeeOnly,
		IOLevelCpuGpuComposite,
		IOLevelProtectedCpuGpuTransfer,
		IOLevelFullDeviceIOAttested,
	}
	for i := 1; i < len(chain); i++ {
		if !(chain[i] > chain[i-1]) {
			t.Fatalf("io level ordering broken at %s -> %s", chain[i-1], chain[i])
		}
	}
}

func TestConfidentialIOLevel_String(t *testing.T) {
	cases := []struct {
		level ConfidentialIOLevel
		want  string
	}{
		{IOLevelNone, "None"},
		{IOLevelCpuTeeOnly, "CpuTeeOnly"},
		{IOLevelCpuGpuComposite, "CpuGpuComposite"},
		{IOLevelProtectedCpuGpuTransfer, "ProtectedCpuGpuTransfer"},
		{IOLevelFullDeviceIOAttested, "FullDeviceIOAttested"},
		{ConfidentialIOLevel(99), "Unknown"},
	}
	for _, c := range cases {
		if got := c.level.String(); got != c.want {
			t.Errorf("%d: got %q want %q", uint8(c.level), got, c.want)
		}
	}
}

func TestConfidentialIOLevel_Valid(t *testing.T) {
	for l := ConfidentialIOLevel(0); l <= IOLevelFullDeviceIOAttested; l++ {
		if !l.Valid() {
			t.Errorf("expected %s to be valid", l)
		}
	}
	if ConfidentialIOLevel(99).Valid() {
		t.Error("expected 99 to be invalid")
	}
}

// TestWorkloadPrivacyClass_StableValues pins the underlying integer values
// of every WorkloadPrivacyClass variant.
func TestWorkloadPrivacyClass_StableValues(t *testing.T) {
	cases := []struct {
		class WorkloadPrivacyClass
		want  uint8
	}{
		{PrivacyPublic, 0},
		{PrivacyPrivateUserData, 1},
		{PrivacyPrivateModelWeights, 2},
		{PrivacyValidatorKeyMaterial, 3},
		{PrivacyRegulatedOrderflow, 4},
		{PrivacyResearchContribution, 5},
	}
	for _, c := range cases {
		if uint8(c.class) != c.want {
			t.Errorf("%s: got %d want %d", c.class, uint8(c.class), c.want)
		}
	}
}

func TestWorkloadPrivacyClass_String(t *testing.T) {
	cases := []struct {
		class WorkloadPrivacyClass
		want  string
	}{
		{PrivacyPublic, "Public"},
		{PrivacyPrivateUserData, "PrivateUserData"},
		{PrivacyPrivateModelWeights, "PrivateModelWeights"},
		{PrivacyValidatorKeyMaterial, "ValidatorKeyMaterial"},
		{PrivacyRegulatedOrderflow, "RegulatedOrderflow"},
		{PrivacyResearchContribution, "ResearchContribution"},
		{WorkloadPrivacyClass(99), "Unknown"},
	}
	for _, c := range cases {
		if got := c.class.String(); got != c.want {
			t.Errorf("%d: got %q want %q", uint8(c.class), got, c.want)
		}
	}
}

func TestWorkloadPrivacyClass_Valid(t *testing.T) {
	for c := WorkloadPrivacyClass(0); c <= PrivacyResearchContribution; c++ {
		if !c.Valid() {
			t.Errorf("expected %s to be valid", c)
		}
	}
	if WorkloadPrivacyClass(99).Valid() {
		t.Error("expected 99 to be invalid")
	}
}

// trustTriple is a Go-side mirror of luxcpp/lattice/test/trust_test.cpp::
// TrustTriple. It is laid out as four bytes so the byte image is identical
// across Go and C++. The only purpose of this struct is the cross-language
// layout test; do not export it.
type trustTriple struct {
	Trust   ComputeTrustMode
	IO      ConfidentialIOLevel
	Privacy WorkloadPrivacyClass
	_pad    uint8
}

// TestTrustEnum_CrossLangByteImage asserts that a {trust, io, privacy, pad}
// quadruple has identical byte representation between Go and C++. The
// matching C++ test is luxcpp/lattice/test/trust_test.cpp::
// test_trust_triple_byte_image. If either side changes enum width or value,
// both this Go test and the C++ memcmp test MUST be updated together.
func TestTrustEnum_CrossLangByteImage(t *testing.T) {
	if got := unsafe.Sizeof(trustTriple{}); got != 4 {
		t.Fatalf("trustTriple size: got %d want 4", got)
	}
	if got := unsafe.Offsetof(trustTriple{}.Trust); got != 0 {
		t.Errorf("Trust offset: got %d want 0", got)
	}
	if got := unsafe.Offsetof(trustTriple{}.IO); got != 1 {
		t.Errorf("IO offset: got %d want 1", got)
	}
	if got := unsafe.Offsetof(trustTriple{}.Privacy); got != 2 {
		t.Errorf("Privacy offset: got %d want 2", got)
	}

	tt := trustTriple{
		Trust:   TrustConfidentialIO,             // 3
		IO:      IOLevelProtectedCpuGpuTransfer,  // 3
		Privacy: PrivacyValidatorKeyMaterial,     // 3
		_pad:    0,
	}
	want := []byte{0x03, 0x03, 0x03, 0x00}
	got := unsafe.Slice((*byte)(unsafe.Pointer(&tt)), unsafe.Sizeof(tt))
	if !bytes.Equal(got, want) {
		t.Fatalf("trustTriple byte image mismatch with C++ side\n  got:  %x\n  want: %x", got, want)
	}
}
