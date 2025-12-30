//go:build !cgo

// Package gpu provides stubs when CGO is disabled.
// Without CGO, GPU acceleration is not available.
package gpu

import "fmt"

// GPUAvailable returns false when CGO is disabled.
func GPUAvailable() bool {
	return false
}

// GetBackend returns "CPU (CGO disabled)".
func GetBackend() string {
	return "CPU (CGO disabled)"
}

// ClearCache is a no-op without CGO.
func ClearCache() {}

// NTTContext is not available without CGO.
type NTTContext struct{}

// NewNTTContext returns an error when CGO is disabled.
func NewNTTContext(N uint32, Q uint64) (*NTTContext, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// Close is a no-op.
func (ctx *NTTContext) Close() {}

// NTT returns an error when CGO is disabled.
func (ctx *NTTContext) NTT(polys [][]uint64) ([][]uint64, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// INTT returns an error when CGO is disabled.
func (ctx *NTTContext) INTT(polys [][]uint64) ([][]uint64, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// PolyMul returns an error when CGO is disabled.
func (ctx *NTTContext) PolyMul(a, b [][]uint64) ([][]uint64, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// PolyMulNTT returns an error when CGO is disabled.
func (ctx *NTTContext) PolyMulNTT(a, b []uint64) ([]uint64, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// PolyAdd returns an error when CGO is disabled.
func PolyAdd(a, b []uint64, Q uint64) ([]uint64, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// PolySub returns an error when CGO is disabled.
func PolySub(a, b []uint64, Q uint64) ([]uint64, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// PolyScalarMul returns an error when CGO is disabled.
func PolyScalarMul(a []uint64, scalar, Q uint64) ([]uint64, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// SampleGaussian returns an error when CGO is disabled.
func SampleGaussian(N uint32, Q uint64, sigma float64, seed []byte) ([]uint64, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// SampleUniform returns an error when CGO is disabled.
func SampleUniform(N uint32, Q uint64, seed []byte) ([]uint64, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// SampleTernary returns an error when CGO is disabled.
func SampleTernary(N uint32, Q uint64, density float64, seed []byte) ([]uint64, error) {
	return nil, fmt.Errorf("GPU acceleration requires CGO")
}

// FindPrimitiveRoot returns an error when CGO is disabled.
func FindPrimitiveRoot(N uint32, Q uint64) (uint64, error) {
	return 0, fmt.Errorf("GPU acceleration requires CGO")
}

// ModInverse returns an error when CGO is disabled.
func ModInverse(a, Q uint64) (uint64, error) {
	return 0, fmt.Errorf("GPU acceleration requires CGO")
}

// IsNTTPrime returns false when CGO is disabled.
func IsNTTPrime(N uint32, Q uint64) bool {
	return false
}
