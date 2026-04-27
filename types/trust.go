// Copyright (c) 2026, Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause

package types

// Trust + privacy enums for the Lux GPU Runtime capability registry.
//
// Reference: LP-137-TRUST-REGISTRY.md.
//
// These enums are the single source of truth for ComputeTrustMode,
// ConfidentialIOLevel, and WorkloadPrivacyClass across the Lux stack:
//
//	luxfi/lattice/types        -- canonical Go definitions (this file)
//	luxfi/mpc/pkg/registry     -- worker / node capability records that
//	                              embed these enums
//	luxfi/mpc/pkg/attestation  -- composite attestation envelopes that
//	                              carry an IOLevel field
//	luxcpp/lattice/.../trust   -- byte-stable C++ mirror
//
// The enums encode a *partial order* over trust strength. A worker with
// trust mode T satisfies a workload requiring minimum trust mode M iff
// `T >= M` (numerical comparison on the underlying uint8). The same is
// true for ConfidentialIOLevel.
//
// Adding a new variant in the middle of the order is a breaking change:
// any downstream Eligible() check would silently flip. New variants MUST
// be appended at the end of the order, and the C++ mirror MUST be updated
// in the same change.

// ComputeTrustMode classifies how strongly a worker's GPU/CPU compute
// path is integrity-attested. Higher values impose strictly stronger
// guarantees than lower values.
//
// Ordering (weak to strong):
//
//   - TrustPublicDeterministic: no privacy guarantees, public inputs +
//     public weights, plaintext device IO. Maximum throughput.
//
//   - TrustAttestedGpuOnly: GPU integrity is attested via a vendor
//     attestation report (e.g. NVIDIA NRAS), but data on the device is
//     not protected. Suitable for public-data + integrity-bound inference.
//
//   - TrustCpuGpuCompositeTEE: CPU TEE (SEV-SNP / TDX) and GPU TEE
//     (Hopper CC, Blackwell CC) are jointly attested. Private workloads
//     run inside the composite TEE; the device-to-device path is still
//     plaintext unless paired with a TrustConfidentialIO IOLevel.
//
//   - TrustConfidentialIO: in addition to composite TEE, the platform
//     enforces a protected IO path between CPU and GPU (and, where
//     present, NVSwitch / DPU). Models + inputs travel encrypted across
//     the PCIe boundary.
//
//   - TrustZKOrFraudProofed: instead of relying on TEE attestation, the
//     workload outputs a zero-knowledge proof or a fraud-proof commitment
//     that lets a verifier check the result without trusting the worker
//     hardware. Reserved for verifiable-compute lanes (LP-025 fraud
//     proofs / LP-013 §ZK).
type ComputeTrustMode uint8

const (
	// TrustPublicDeterministic: no privacy, max throughput.
	TrustPublicDeterministic ComputeTrustMode = 0

	// TrustAttestedGpuOnly: GPU integrity attested, data public.
	TrustAttestedGpuOnly ComputeTrustMode = 1

	// TrustCpuGpuCompositeTEE: CPU TEE + GPU TEE jointly attested.
	TrustCpuGpuCompositeTEE ComputeTrustMode = 2

	// TrustConfidentialIO: protected IO path where the platform supports
	// it (Hopper / Blackwell CC + protected PCIe / NVSwitch).
	TrustConfidentialIO ComputeTrustMode = 3

	// TrustZKOrFraudProofed: cryptographic proof or fraud-proof check
	// substitutes for hardware attestation.
	TrustZKOrFraudProofed ComputeTrustMode = 4
)

// String returns a stable, human-readable name for the trust mode.
// The string values are part of the wire/log contract; do not rename.
func (m ComputeTrustMode) String() string {
	switch m {
	case TrustPublicDeterministic:
		return "PublicDeterministic"
	case TrustAttestedGpuOnly:
		return "AttestedGpuOnly"
	case TrustCpuGpuCompositeTEE:
		return "CpuGpuCompositeTEE"
	case TrustConfidentialIO:
		return "ConfidentialIO"
	case TrustZKOrFraudProofed:
		return "ZKOrFraudProofed"
	default:
		return "Unknown"
	}
}

// Valid reports whether m is one of the defined trust modes.
func (m ComputeTrustMode) Valid() bool {
	return m <= TrustZKOrFraudProofed
}

// ConfidentialIOLevel classifies the integrity + confidentiality of the
// data path between host CPU, GPU, and device peers (NVSwitch, DPU NIC).
// Higher values impose strictly stronger guarantees than lower values.
//
// Ordering (weak to strong):
//
//   - IOLevelNone: no protection, plaintext IO.
//   - IOLevelCpuTeeOnly: CPU TEE protects host memory; GPU IO unprotected.
//   - IOLevelCpuGpuComposite: CPU TEE + GPU TEE both attested; device-side
//     buffers protected against host inspection.
//   - IOLevelProtectedCpuGpuTransfer: PCIe transfers between CPU TEE and
//     GPU TEE are encrypted (or bus-encrypted via Hopper CC bounce).
//   - IOLevelFullDeviceIOAttested: full attested device IO path including
//     NVSwitch + DPU NIC encryption. Required for multi-GPU confidential
//     workloads with cross-host fan-out.
type ConfidentialIOLevel uint8

const (
	// IOLevelNone: no IO protection.
	IOLevelNone ConfidentialIOLevel = 0

	// IOLevelCpuTeeOnly: CPU TEE only.
	IOLevelCpuTeeOnly ConfidentialIOLevel = 1

	// IOLevelCpuGpuComposite: CPU TEE + GPU TEE attested, device buffers
	// protected.
	IOLevelCpuGpuComposite ConfidentialIOLevel = 2

	// IOLevelProtectedCpuGpuTransfer: encrypted CPU<->GPU PCIe transfer.
	IOLevelProtectedCpuGpuTransfer ConfidentialIOLevel = 3

	// IOLevelFullDeviceIOAttested: full attested device IO path
	// (NVSwitch + DPU NIC included).
	IOLevelFullDeviceIOAttested ConfidentialIOLevel = 4
)

// String returns a stable name for the IO level.
func (l ConfidentialIOLevel) String() string {
	switch l {
	case IOLevelNone:
		return "None"
	case IOLevelCpuTeeOnly:
		return "CpuTeeOnly"
	case IOLevelCpuGpuComposite:
		return "CpuGpuComposite"
	case IOLevelProtectedCpuGpuTransfer:
		return "ProtectedCpuGpuTransfer"
	case IOLevelFullDeviceIOAttested:
		return "FullDeviceIOAttested"
	default:
		return "Unknown"
	}
}

// Valid reports whether l is one of the defined IO levels.
func (l ConfidentialIOLevel) Valid() bool {
	return l <= IOLevelFullDeviceIOAttested
}

// WorkloadPrivacyClass classifies the data-handling sensitivity of a
// workload. Privacy class drives the *minimum* trust mode + IO level the
// scheduler may match against. The mapping from class to (MinTrustMode,
// MinIOLevel) lives in WorkloadPolicy and LanePolicy, NOT here -- this
// enum is purely a tag.
//
// The order has no operational meaning and is kept stable for log /
// telemetry parity across Go and C++.
type WorkloadPrivacyClass uint8

const (
	// PrivacyPublic: public inputs, public weights, public outputs.
	PrivacyPublic WorkloadPrivacyClass = 0

	// PrivacyPrivateUserData: user-provided inputs are private; weights
	// public.
	PrivacyPrivateUserData WorkloadPrivacyClass = 1

	// PrivacyPrivateModelWeights: model weights are private; inputs may
	// or may not be private.
	PrivacyPrivateModelWeights WorkloadPrivacyClass = 2

	// PrivacyValidatorKeyMaterial: workload touches validator-grade key
	// material (signing, MPC share custody). Highest possible isolation
	// requirement.
	PrivacyValidatorKeyMaterial WorkloadPrivacyClass = 3

	// PrivacyRegulatedOrderflow: regulated order flow (ATS / BD / TA) --
	// confidentiality is mandated by jurisdiction, not just policy.
	PrivacyRegulatedOrderflow WorkloadPrivacyClass = 4

	// PrivacyResearchContribution: federated research contribution where
	// gradients / activations must be hidden from coordinator and peers.
	PrivacyResearchContribution WorkloadPrivacyClass = 5
)

// String returns a stable name for the privacy class.
func (c WorkloadPrivacyClass) String() string {
	switch c {
	case PrivacyPublic:
		return "Public"
	case PrivacyPrivateUserData:
		return "PrivateUserData"
	case PrivacyPrivateModelWeights:
		return "PrivateModelWeights"
	case PrivacyValidatorKeyMaterial:
		return "ValidatorKeyMaterial"
	case PrivacyRegulatedOrderflow:
		return "RegulatedOrderflow"
	case PrivacyResearchContribution:
		return "ResearchContribution"
	default:
		return "Unknown"
	}
}

// Valid reports whether c is one of the defined privacy classes.
func (c WorkloadPrivacyClass) Valid() bool {
	return c <= PrivacyResearchContribution
}
