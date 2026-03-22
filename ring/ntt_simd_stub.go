//go:build !(goexperiment.simd && amd64)

package ring

// nttCoreLazyAccel is the stub for non-SIMD builds.
// Returns false to indicate no acceleration was applied.
func nttCoreLazyAccel(p1, p2 []uint64, N int, Q, MRedConstant uint64, roots []uint64) bool {
	return false
}

// inttCoreLazyAccel is the stub for non-SIMD builds.
// Returns false to indicate no acceleration was applied.
func inttCoreLazyAccel(p1, p2 []uint64, N int, Q, MRedConstant uint64, roots []uint64) bool {
	return false
}
