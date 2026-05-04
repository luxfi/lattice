//go:build !(goexperiment.simd && amd64)

package ring

// SIMD dispatch hooks for non-AVX2-amd64 builds.
//
// On builds without GOEXPERIMENT=simd or off-amd64, the AVX2 NTT path
// is unavailable. These hooks return false so that nttCoreLazy /
// inttCoreLazy (ntt.go:217, ntt.go:567) falls through to the
// canonical pure-Go scalar NTT (nttLazy / nttUnrolled16Lazy /
// inttLazy / inttUnrolled16Lazy in this same package).
//
// The pure-Go scalar NTT is the canonical implementation; SIMD is a
// strict speed-up on top of it, never a correctness boundary. The
// `_purego` suffix matches the rest of the build-tag pattern used in
// gpu/gpu_montgomery_purego.go.

// nttCoreLazyAccel returns false on non-SIMD builds; the caller runs
// the canonical pure-Go scalar NTT path.
func nttCoreLazyAccel(p1, p2 []uint64, N int, Q, MRedConstant uint64, roots []uint64) bool {
	return false
}

// inttCoreLazyAccel returns false on non-SIMD builds; the caller runs
// the canonical pure-Go scalar INTT path.
func inttCoreLazyAccel(p1, p2 []uint64, N int, Q, MRedConstant uint64, roots []uint64) bool {
	return false
}
