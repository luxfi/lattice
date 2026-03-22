//go:build goexperiment.simd && amd64

package ring

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

// simdAVX2 reports whether the CPU supports AVX2 (minimum for SIMD NTT).
var simdAVX2 = archsimd.X86.AVX2()

// mredLazy4 computes 4 parallel MRedLazy operations using AVX2 VPMULUDQ.
//
// MRedLazy(x, y, Q, MRedConstant) computes:
//
//	ahi, alo = x * y                       (128-bit multiply)
//	H, _    = (alo * MRedConstant) * Q    (128-bit multiply, high 64 bits)
//	result  = ahi - H + Q
//
// 64x64->128 bit multiply is decomposed into 32-bit widening multiplies:
//
//	a*b = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)<<32 + a_hi*b_hi<<64
//
// We use Uint32x8.MulEvenWiden (VPMULUDQ, AVX2) which multiplies even-indexed
// 32-bit elements producing 64-bit results. By reinterpreting Uint64x4 as
// Uint32x8, the even positions hold the low 32 bits of each uint64.
//
//go:nosplit
func mredLazy4(xp, yp *[4]uint64, Q, MRedConstant uint64) archsimd.Uint64x4 {
	xv := archsimd.LoadUint64x4(xp)
	yv := archsimd.LoadUint64x4(yp)
	qv := archsimd.BroadcastUint64x4(Q)
	mredv := archsimd.BroadcastUint64x4(MRedConstant)

	ahi, alo := mul64x4(xv, yv)

	// t = alo * MRedConstant (low 64 bits only)
	t := mulLow64x4(alo, mredv)

	// H = high 64 bits of (t * Q)
	H, _ := mul64x4(t, qv)

	// result = ahi - H + Q
	return ahi.Sub(H).Add(qv)
}

// mul64x4 computes 4 parallel 64x64->128 multiplies, returning high and low 64-bit halves.
// Uses VPMULUDQ (AVX2) decomposition into 32-bit widening multiplies.
//
//go:nosplit
func mul64x4(a, b archsimd.Uint64x4) (hi, lo archsimd.Uint64x4) {
	// Decompose: a = a_hi<<32 | a_lo, b = b_hi<<32 | b_lo
	a32 := a.AsUint32x8() // even indices = lo32, odd indices = hi32
	b32 := b.AsUint32x8()

	// VPMULUDQ: multiplies even-indexed 32-bit elements -> 64-bit results
	// p0[i] = a_lo[i] * b_lo[i]  (4 x 64-bit results)
	p0 := a32.MulEvenWiden(b32)

	// For cross terms, shift to get hi32 into even positions
	aHi := a.ShiftAllRight(32).AsUint32x8()
	bHi := b.ShiftAllRight(32).AsUint32x8()

	// p1[i] = a_lo[i] * b_hi[i]
	p1 := a32.MulEvenWiden(bHi)
	// p2[i] = a_hi[i] * b_lo[i]
	p2 := aHi.MulEvenWiden(b32)
	// p3[i] = a_hi[i] * b_hi[i]
	p3 := aHi.MulEvenWiden(bHi)

	// Combine: result = p0 + (p1 + p2)<<32 + p3<<64
	// lo = p0 + (p1 + p2)<<32  (mod 2^64)
	// hi = p3 + (p1 + p2)>>32 + carries

	mid := p1.Add(p2)

	// Carry from mid addition: if mid < p1, overflow occurred (adds 1<<32 to hi)
	midCarry := mid.Less(p1)
	midCarryVal := archsimd.BroadcastUint64x4(1 << 32).Masked(midCarry)

	lo = p0.Add(mid.ShiftAllLeft(32))

	// Carry from lo addition: if lo < p0, overflow occurred (adds 1 to hi)
	loCarry := lo.Less(p0)
	loCarryVal := archsimd.BroadcastUint64x4(1).Masked(loCarry)

	hi = p3.Add(mid.ShiftAllRight(32)).Add(midCarryVal).Add(loCarryVal)

	return hi, lo
}

// mulLow64x4 computes 4 parallel 64x64 multiplies, returning only the low 64 bits.
// Uses VPMULUDQ (AVX2) decomposition.
//
//go:nosplit
func mulLow64x4(a, b archsimd.Uint64x4) archsimd.Uint64x4 {
	a32 := a.AsUint32x8()
	b32 := b.AsUint32x8()

	// p0 = a_lo * b_lo (full 64-bit, but we only need low 64 of final result)
	p0 := a32.MulEvenWiden(b32)

	// Cross terms: only the low 32 bits of each contribute to the final low 64
	aHi := a.ShiftAllRight(32).AsUint32x8()
	bHi := b.ShiftAllRight(32).AsUint32x8()

	p1 := a32.MulEvenWiden(bHi) // a_lo * b_hi
	p2 := aHi.MulEvenWiden(b32) // a_hi * b_lo

	// lo = p0 + (p1 + p2) << 32  (mod 2^64)
	return p0.Add(p1.Add(p2).ShiftAllLeft(32))
}

// nttCoreLazyAccel attempts SIMD-accelerated NTT.
// Returns true if acceleration was applied, false to fall back to scalar.
func nttCoreLazyAccel(p1, p2 []uint64, N int, Q, MRedConstant uint64, roots []uint64) bool {
	if !simdAVX2 || N < MinimumRingDegreeForLoopUnrolledNTT {
		return false
	}

	var j1, j2, t int
	var F uint64

	fourQ := 4 * Q
	twoQ := 2 * Q

	// First round: copy p1 -> p2 with butterfly, using SIMD for 4-wide MRedLazy
	t = N >> 1
	F = roots[1]
	psiArr := [4]uint64{F, F, F, F}
	twoQv := archsimd.BroadcastUint64x4(twoQ)

	for jx, jy := 0, t; jx < t; jx, jy = jx+4, jy+4 {
		/* #nosec G103 -- aligned 4-element access within bounds */
		xin := (*[4]uint64)(unsafe.Pointer(&p1[jx]))
		yin := (*[4]uint64)(unsafe.Pointer(&p1[jy]))

		V := mredLazy4(yin, &psiArr, Q, MRedConstant)
		xv := archsimd.LoadUint64x4(xin)

		xv.Add(V).StoreSlice(p2[jx:])
		xv.Add(twoQv).Sub(V).StoreSlice(p2[jy:])
	}

	// Remaining stages
	var reduce bool
	for m := 2; m < N; m <<= 1 {
		/* #nosec G115 -- m cannot be negative */
		reduce = (bits.Len64(uint64(m))&1 == 1)
		t >>= 1

		if t >= 8 {
			for i := 0; i < m; i++ {
				j1 = (i * t) << 1
				j2 = j1 + t
				F = roots[m+i]
				psiArr = [4]uint64{F, F, F, F}
				fourQv := archsimd.BroadcastUint64x4(fourQ)

				if reduce {
					for jx, jy := j1, j1+t; jx < j2; jx, jy = jx+4, jy+4 {
						xp := (*[4]uint64)(unsafe.Pointer(&p2[jx]))
						yp := (*[4]uint64)(unsafe.Pointer(&p2[jy]))

						xv := archsimd.LoadUint64x4(xp)
						// Conditional reduction: if U >= fourQ, U -= fourQ
						needReduce := xv.GreaterEqual(fourQv)
						xv = xv.Sub(fourQv.Masked(needReduce))

						V := mredLazy4(yp, &psiArr, Q, MRedConstant)
						xout := xv.Add(V)
						yout := xv.Add(twoQv).Sub(V)

						xout.Store(xp)
						yout.Store(yp)
					}
				} else {
					for jx, jy := j1, j1+t; jx < j2; jx, jy = jx+4, jy+4 {
						xp := (*[4]uint64)(unsafe.Pointer(&p2[jx]))
						yp := (*[4]uint64)(unsafe.Pointer(&p2[jy]))

						xv := archsimd.LoadUint64x4(xp)
						V := mredLazy4(yp, &psiArr, Q, MRedConstant)
						xout := xv.Add(V)
						yout := xv.Add(twoQv).Sub(V)

						xout.Store(xp)
						yout.Store(yp)
					}
				}
			}
		} else {
			// Small t stages: fall back to scalar (different memory access patterns)
			nttSmallTStagesScalar(p2, m, t, Q, MRedConstant, twoQ, fourQ, roots, reduce)
		}
	}

	return true
}

// inttCoreLazyAccel attempts SIMD-accelerated INTT.
// Returns true if acceleration was applied, false to fall back to scalar.
func inttCoreLazyAccel(p1, p2 []uint64, N int, Q, MRedConstant uint64, roots []uint64) bool {
	if !simdAVX2 || N < MinimumRingDegreeForLoopUnrolledNTT {
		return false
	}

	var h, t int
	var F uint64

	t = 1
	h = N >> 1
	twoQ := Q << 1
	fourQ := Q << 2

	// First round (t=1): scalar, different access pattern per butterfly
	for i, j := h, 0; i < 2*h; i, j = i+8, j+16 {
		/* #nosec G103 -- behavior and consequences well understood */
		psi := (*[8]uint64)(unsafe.Pointer(&roots[i]))
		xin := (*[16]uint64)(unsafe.Pointer(&p1[j]))
		xout := (*[16]uint64)(unsafe.Pointer(&p2[j]))

		xout[0], xout[1] = invbutterfly(xin[0], xin[1], psi[0], twoQ, fourQ, Q, MRedConstant)
		xout[2], xout[3] = invbutterfly(xin[2], xin[3], psi[1], twoQ, fourQ, Q, MRedConstant)
		xout[4], xout[5] = invbutterfly(xin[4], xin[5], psi[2], twoQ, fourQ, Q, MRedConstant)
		xout[6], xout[7] = invbutterfly(xin[6], xin[7], psi[3], twoQ, fourQ, Q, MRedConstant)
		xout[8], xout[9] = invbutterfly(xin[8], xin[9], psi[4], twoQ, fourQ, Q, MRedConstant)
		xout[10], xout[11] = invbutterfly(xin[10], xin[11], psi[5], twoQ, fourQ, Q, MRedConstant)
		xout[12], xout[13] = invbutterfly(xin[12], xin[13], psi[6], twoQ, fourQ, Q, MRedConstant)
		xout[14], xout[15] = invbutterfly(xin[14], xin[15], psi[7], twoQ, fourQ, Q, MRedConstant)
	}

	t <<= 1
	twoQv := archsimd.BroadcastUint64x4(twoQ)
	fourQv := archsimd.BroadcastUint64x4(fourQ)
	for m := N >> 1; m > 1; m >>= 1 {
		h = m >> 1

		if t >= 8 {
			for i, j1, j2 := 0, 0, t; i < h; i, j1, j2 = i+1, j1+2*t, j2+2*t {
				F = roots[h+i]
				psiArr := [4]uint64{F, F, F, F}

				for jx, jy := j1, j1+t; jx < j2; jx, jy = jx+4, jy+4 {
					xp := (*[4]uint64)(unsafe.Pointer(&p2[jx]))
					yp := (*[4]uint64)(unsafe.Pointer(&p2[jy]))

					xv := archsimd.LoadUint64x4(xp)
					yv := archsimd.LoadUint64x4(yp)

					// X = U + V; if X >= twoQ, X -= twoQ
					sum := xv.Add(yv)
					needReduce := sum.GreaterEqual(twoQv)
					xout := sum.Sub(twoQv.Masked(needReduce))

					// Y = MRedLazy(U + fourQ - V, Psi, Q, MRedConstant)
					diff := xv.Add(fourQv).Sub(yv)
					var diffArr [4]uint64
					diff.Store(&diffArr)
					yout := mredLazy4(&diffArr, &psiArr, Q, MRedConstant)

					xout.Store(xp)
					yout.Store(yp)
				}
			}
		} else if t == 4 {
			for i, j1 := h, 0; i < 2*h; i, j1 = i+2, j1+4*t {
				/* #nosec G103 -- behavior and consequences well understood */
				psi := (*[2]uint64)(unsafe.Pointer(&roots[i]))
				x := (*[16]uint64)(unsafe.Pointer(&p2[j1]))
				x[0], x[4] = invbutterfly(x[0], x[4], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[1], x[5] = invbutterfly(x[1], x[5], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[2], x[6] = invbutterfly(x[2], x[6], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[3], x[7] = invbutterfly(x[3], x[7], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[8], x[12] = invbutterfly(x[8], x[12], psi[1], twoQ, fourQ, Q, MRedConstant)
				x[9], x[13] = invbutterfly(x[9], x[13], psi[1], twoQ, fourQ, Q, MRedConstant)
				x[10], x[14] = invbutterfly(x[10], x[14], psi[1], twoQ, fourQ, Q, MRedConstant)
				x[11], x[15] = invbutterfly(x[11], x[15], psi[1], twoQ, fourQ, Q, MRedConstant)
			}
		} else {
			for i, j1 := h, 0; i < 2*h; i, j1 = i+4, j1+8*t {
				/* #nosec G103 -- behavior and consequences well understood */
				psi := (*[4]uint64)(unsafe.Pointer(&roots[i]))
				x := (*[16]uint64)(unsafe.Pointer(&p2[j1]))
				x[0], x[2] = invbutterfly(x[0], x[2], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[1], x[3] = invbutterfly(x[1], x[3], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[4], x[6] = invbutterfly(x[4], x[6], psi[1], twoQ, fourQ, Q, MRedConstant)
				x[5], x[7] = invbutterfly(x[5], x[7], psi[1], twoQ, fourQ, Q, MRedConstant)
				x[8], x[10] = invbutterfly(x[8], x[10], psi[2], twoQ, fourQ, Q, MRedConstant)
				x[9], x[11] = invbutterfly(x[9], x[11], psi[2], twoQ, fourQ, Q, MRedConstant)
				x[12], x[14] = invbutterfly(x[12], x[14], psi[3], twoQ, fourQ, Q, MRedConstant)
				x[13], x[15] = invbutterfly(x[13], x[15], psi[3], twoQ, fourQ, Q, MRedConstant)
			}
		}

		t <<= 1
	}

	return true
}

// nttSmallTStagesScalar handles NTT butterfly stages where t < 8.
// These stages use non-contiguous memory access patterns (stride != 1)
// that don't benefit from 4-wide contiguous SIMD loads/stores.
func nttSmallTStagesScalar(p2 []uint64, m, t int, Q, MRedConstant, twoQ, fourQ uint64, roots []uint64, reduce bool) {
	var V uint64

	if t == 4 {
		if reduce {
			for i, j1 := m, 0; i < 2*m; i, j1 = i+2, j1+4*t {
				/* #nosec G103 -- behavior and consequences well understood */
				psi := (*[2]uint64)(unsafe.Pointer(&roots[i]))
				x := (*[16]uint64)(unsafe.Pointer(&p2[j1]))
				x[0], x[4] = butterfly(x[0], x[4], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[1], x[5] = butterfly(x[1], x[5], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[2], x[6] = butterfly(x[2], x[6], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[3], x[7] = butterfly(x[3], x[7], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[8], x[12] = butterfly(x[8], x[12], psi[1], twoQ, fourQ, Q, MRedConstant)
				x[9], x[13] = butterfly(x[9], x[13], psi[1], twoQ, fourQ, Q, MRedConstant)
				x[10], x[14] = butterfly(x[10], x[14], psi[1], twoQ, fourQ, Q, MRedConstant)
				x[11], x[15] = butterfly(x[11], x[15], psi[1], twoQ, fourQ, Q, MRedConstant)
			}
		} else {
			for i, j1 := m, 0; i < 2*m; i, j1 = i+2, j1+4*t {
				/* #nosec G103 -- behavior and consequences well understood */
				psi := (*[2]uint64)(unsafe.Pointer(&roots[i]))
				x := (*[16]uint64)(unsafe.Pointer(&p2[j1]))
				V = MRedLazy(x[4], psi[0], Q, MRedConstant)
				x[0], x[4] = x[0]+V, x[0]+twoQ-V
				V = MRedLazy(x[5], psi[0], Q, MRedConstant)
				x[1], x[5] = x[1]+V, x[1]+twoQ-V
				V = MRedLazy(x[6], psi[0], Q, MRedConstant)
				x[2], x[6] = x[2]+V, x[2]+twoQ-V
				V = MRedLazy(x[7], psi[0], Q, MRedConstant)
				x[3], x[7] = x[3]+V, x[3]+twoQ-V
				V = MRedLazy(x[12], psi[1], Q, MRedConstant)
				x[8], x[12] = x[8]+V, x[8]+twoQ-V
				V = MRedLazy(x[13], psi[1], Q, MRedConstant)
				x[9], x[13] = x[9]+V, x[9]+twoQ-V
				V = MRedLazy(x[14], psi[1], Q, MRedConstant)
				x[10], x[14] = x[10]+V, x[10]+twoQ-V
				V = MRedLazy(x[15], psi[1], Q, MRedConstant)
				x[11], x[15] = x[11]+V, x[11]+twoQ-V
			}
		}
	} else if t == 2 {
		if reduce {
			for i, j1 := m, 0; i < 2*m; i, j1 = i+4, j1+8*t {
				/* #nosec G103 -- behavior and consequences well understood */
				psi := (*[4]uint64)(unsafe.Pointer(&roots[i]))
				x := (*[16]uint64)(unsafe.Pointer(&p2[j1]))
				x[0], x[2] = butterfly(x[0], x[2], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[1], x[3] = butterfly(x[1], x[3], psi[0], twoQ, fourQ, Q, MRedConstant)
				x[4], x[6] = butterfly(x[4], x[6], psi[1], twoQ, fourQ, Q, MRedConstant)
				x[5], x[7] = butterfly(x[5], x[7], psi[1], twoQ, fourQ, Q, MRedConstant)
				x[8], x[10] = butterfly(x[8], x[10], psi[2], twoQ, fourQ, Q, MRedConstant)
				x[9], x[11] = butterfly(x[9], x[11], psi[2], twoQ, fourQ, Q, MRedConstant)
				x[12], x[14] = butterfly(x[12], x[14], psi[3], twoQ, fourQ, Q, MRedConstant)
				x[13], x[15] = butterfly(x[13], x[15], psi[3], twoQ, fourQ, Q, MRedConstant)
			}
		} else {
			for i, j1 := m, 0; i < 2*m; i, j1 = i+4, j1+8*t {
				/* #nosec G103 -- behavior and consequences well understood */
				psi := (*[4]uint64)(unsafe.Pointer(&roots[i]))
				x := (*[16]uint64)(unsafe.Pointer(&p2[j1]))
				V = MRedLazy(x[2], psi[0], Q, MRedConstant)
				x[0], x[2] = x[0]+V, x[0]+twoQ-V
				V = MRedLazy(x[3], psi[0], Q, MRedConstant)
				x[1], x[3] = x[1]+V, x[1]+twoQ-V
				V = MRedLazy(x[6], psi[1], Q, MRedConstant)
				x[4], x[6] = x[4]+V, x[4]+twoQ-V
				V = MRedLazy(x[7], psi[1], Q, MRedConstant)
				x[5], x[7] = x[5]+V, x[5]+twoQ-V
				V = MRedLazy(x[10], psi[2], Q, MRedConstant)
				x[8], x[10] = x[8]+V, x[8]+twoQ-V
				V = MRedLazy(x[11], psi[2], Q, MRedConstant)
				x[9], x[11] = x[9]+V, x[9]+twoQ-V
				V = MRedLazy(x[14], psi[3], Q, MRedConstant)
				x[12], x[14] = x[12]+V, x[12]+twoQ-V
				V = MRedLazy(x[15], psi[3], Q, MRedConstant)
				x[13], x[15] = x[13]+V, x[13]+twoQ-V
			}
		}
	} else {
		// t == 1
		for i, j1 := m, 0; i < 2*m; i, j1 = i+8, j1+16 {
			/* #nosec G103 -- behavior and consequences well understood */
			psi := (*[8]uint64)(unsafe.Pointer(&roots[i]))
			x := (*[16]uint64)(unsafe.Pointer(&p2[j1]))
			x[0], x[1] = butterfly(x[0], x[1], psi[0], twoQ, fourQ, Q, MRedConstant)
			x[2], x[3] = butterfly(x[2], x[3], psi[1], twoQ, fourQ, Q, MRedConstant)
			x[4], x[5] = butterfly(x[4], x[5], psi[2], twoQ, fourQ, Q, MRedConstant)
			x[6], x[7] = butterfly(x[6], x[7], psi[3], twoQ, fourQ, Q, MRedConstant)
			x[8], x[9] = butterfly(x[8], x[9], psi[4], twoQ, fourQ, Q, MRedConstant)
			x[10], x[11] = butterfly(x[10], x[11], psi[5], twoQ, fourQ, Q, MRedConstant)
			x[12], x[13] = butterfly(x[12], x[13], psi[6], twoQ, fourQ, Q, MRedConstant)
			x[14], x[15] = butterfly(x[14], x[15], psi[7], twoQ, fourQ, Q, MRedConstant)
		}
	}
}
