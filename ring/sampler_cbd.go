package ring

import (
	"encoding/binary"

	"github.com/luxfi/lattice/v7/utils/sampling"
)

// CBDSampler keeps the state of a constant-time Centered Binomial Distribution
// polynomial sampler. This is the sampling method used by NIST ML-KEM (Kyber)
// and provides constant-time operation with no data-dependent branching.
//
// For parameter Eta, each coefficient is computed as:
//
//	a = popcount(random η bits)
//	b = popcount(random η bits)
//	coefficient = a - b
//
// This produces coefficients in [-Eta, Eta] with a binomial distribution
// centered at 0. The number of random bits consumed per coefficient is fixed
// at 2*Eta, making the sampler trivially constant-time.
type CBDSampler struct {
	*baseSampler
	eta        int
	montgomery bool
}

// NewCBDSampler creates a new instance of CBDSampler from a PRNG, ring definition,
// and CBD distribution parameters.
// WARNING: If the PRNG is deterministic/keyed (of type [sampling.KeyedPRNG]),
// *concurrent* calls to the sampler will not necessarily result in a deterministic output.
func NewCBDSampler(prng sampling.PRNG, baseRing *Ring, X CenteredBinomial, montgomery bool) *CBDSampler {
	return &CBDSampler{
		baseSampler: &baseSampler{
			prng:     prng,
			baseRing: baseRing,
		},
		eta:        X.Eta,
		montgomery: montgomery,
	}
}

// AtLevel returns an instance of the target CBDSampler that operates at the target level.
// This instance is not thread safe and cannot be used concurrently to the base instance.
func (s *CBDSampler) AtLevel(level int) Sampler {
	return &CBDSampler{
		baseSampler: s.baseSampler.AtLevel(level),
		eta:         s.eta,
		montgomery:  s.montgomery,
	}
}

// Read samples a constant-time CBD polynomial on "pol" at the maximum level.
func (s *CBDSampler) Read(pol Poly) {
	s.read(pol, func(a, b, c uint64) uint64 {
		return b
	})
}

// ReadNew samples a new constant-time CBD polynomial at the maximum level.
func (s *CBDSampler) ReadNew() (pol Poly) {
	pol = s.baseRing.NewPoly()
	s.Read(pol)
	return pol
}

// ReadAndAdd samples a constant-time CBD polynomial and adds it on "pol".
func (s *CBDSampler) ReadAndAdd(pol Poly) {
	s.read(pol, func(a, b, c uint64) uint64 {
		return CRed(a+b, c)
	})
}

// read is the core sampling routine. It consumes exactly 2*eta bits per
// coefficient with no data-dependent branching, making it constant-time.
func (s *CBDSampler) read(pol Poly, f func(a, b, c uint64) uint64) {
	r := s.baseRing
	N := r.N()
	eta := s.eta
	moduli := r.ModuliChain()[:r.level+1]
	coeffs := pol.Coeffs

	// Bytes needed: 2 * eta bits per coefficient, N coefficients
	// = (2 * eta * N) / 8 bytes
	numBytes := (2 * eta * N) / 8
	if (2*eta*N)%8 != 0 {
		numBytes++
	}

	buf := make([]byte, numBytes)
	if _, err := s.prng.Read(buf); err != nil {
		panic(err)
	}

	switch eta {
	case 2:
		s.cbdEta2(coeffs, moduli, N, buf, f)
	case 3:
		s.cbdEta3(coeffs, moduli, N, buf, f)
	default:
		s.cbdGeneric(coeffs, moduli, N, buf, f)
	}

	if s.montgomery {
		r.MForm(pol, pol)
	}
}

// cbdEta2 is an optimized path for eta=2 (ML-KEM-512/768).
// Each coefficient uses 4 bits: coeff = popcount(bits[0:2]) - popcount(bits[2:4]).
// Processes 16 coefficients per uint64 (64 bits / 4 bits per coeff).
func (s *CBDSampler) cbdEta2(coeffs [][]uint64, moduli []uint64, N int, buf []byte, f func(a, b, c uint64) uint64) {
	for i := 0; i < N; i += 16 {
		// Read 8 bytes = 64 bits = 16 coefficients at 4 bits each
		offset := i / 2 // 4 bits per coeff = 0.5 bytes
		var d uint64
		if offset+8 <= len(buf) {
			d = binary.LittleEndian.Uint64(buf[offset : offset+8])
		}

		for k := 0; k < 16 && i+k < N; k++ {
			// Extract 4 bits for this coefficient
			bits4 := (d >> (k * 4)) & 0xF

			// a = popcount of lower 2 bits, b = popcount of upper 2 bits
			a := (bits4 & 1) + ((bits4 >> 1) & 1)
			b := ((bits4 >> 2) & 1) + ((bits4 >> 3) & 1)

			// coeff = a - b, in range [-2, 2]
			coeffVal := a - b

			for j, qi := range moduli {
				// Convert signed to modular: if negative, add qi
				// Constant-time: use mask instead of branch
				// coeffVal is uint64, so if a < b, it wraps around
				// We need: if a >= b then coeffVal else qi - (b - a)
				mask := ctLt(a, b) // 0xFFFF...F if a < b, 0 otherwise
				pos := a - b       // correct if a >= b
				neg := qi - (b - a)
				val := (pos & ^mask) | (neg & mask)
				coeffs[j][i+k] = f(coeffs[j][i+k], val, qi)
			}

			_ = coeffVal // suppress unused
		}
	}
}

// cbdEta3 is an optimized path for eta=3.
// Each coefficient uses 6 bits: coeff = popcount(bits[0:3]) - popcount(bits[3:6]).
func (s *CBDSampler) cbdEta3(coeffs [][]uint64, moduli []uint64, N int, buf []byte, f func(a, b, c uint64) uint64) {
	bitPos := 0
	for i := 0; i < N; i++ {
		// Extract 6 bits starting at bitPos
		byteIdx := bitPos / 8
		/* #nosec G115 -- bitPos%8 is always in [0,7] */
		bitOff := uint(bitPos % 8)

		var bits6 uint64
		if byteIdx+2 < len(buf) {
			bits6 = (uint64(buf[byteIdx]) | uint64(buf[byteIdx+1])<<8 | uint64(buf[byteIdx+2])<<16) >> bitOff
		} else if byteIdx+1 < len(buf) {
			bits6 = (uint64(buf[byteIdx]) | uint64(buf[byteIdx+1])<<8) >> bitOff
		} else {
			bits6 = uint64(buf[byteIdx]) >> bitOff
		}
		bits6 &= 0x3F // 6 bits

		// a = popcount of lower 3 bits, b = popcount of upper 3 bits
		a := (bits6 & 1) + ((bits6 >> 1) & 1) + ((bits6 >> 2) & 1)
		b := ((bits6 >> 3) & 1) + ((bits6 >> 4) & 1) + ((bits6 >> 5) & 1)

		for j, qi := range moduli {
			mask := ctLt(a, b)
			pos := a - b
			neg := qi - (b - a)
			val := (pos & ^mask) | (neg & mask)
			coeffs[j][i] = f(coeffs[j][i], val, qi)
		}

		bitPos += 6
	}
}

// cbdGeneric handles arbitrary eta values.
func (s *CBDSampler) cbdGeneric(coeffs [][]uint64, moduli []uint64, N int, buf []byte, f func(a, b, c uint64) uint64) {
	bitPos := 0
	bitsPerCoeff := 2 * s.eta

	for i := 0; i < N; i++ {
		var a, b uint64

		// Count popcount of first eta bits
		for k := 0; k < s.eta; k++ {
			byteIdx := bitPos / 8
			/* #nosec G115 -- bitPos%8 is always in [0,7] */
			bitOff := uint(bitPos % 8)
			a += (uint64(buf[byteIdx]) >> bitOff) & 1
			bitPos++
		}

		// Count popcount of next eta bits
		for k := 0; k < s.eta; k++ {
			byteIdx := bitPos / 8
			/* #nosec G115 -- bitPos%8 is always in [0,7] */
			bitOff := uint(bitPos % 8)
			b += (uint64(buf[byteIdx]) >> bitOff) & 1
			bitPos++
		}

		for j, qi := range moduli {
			mask := ctLt(a, b)
			pos := a - b
			neg := qi - (b - a)
			val := (pos & ^mask) | (neg & mask)
			coeffs[j][i] = f(coeffs[j][i], val, qi)
		}

		_ = bitsPerCoeff
	}
}

// ctLt returns 0xFFFFFFFFFFFFFFFF if a < b, 0 otherwise (constant-time).
func ctLt(a, b uint64) uint64 {
	// If a < b, then a - b underflows, setting the high bit.
	/* #nosec G115 -- intentional: constant-time comparison requires signed arithmetic on the borrow bit */
	return uint64(int64(a-b) >> 63) //nolint:gosec
}
