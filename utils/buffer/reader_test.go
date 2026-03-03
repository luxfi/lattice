// Copyright (c) Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

package buffer

import (
	"errors"
	"io"
	"testing"
)

// truncReader is a Reader implementation that exposes a payload shorter
// than what its Size() advertises. This mirrors the malicious wire format
// discovered by luxfi/warp/pulsar's FuzzPulseDeserialize: a giant outer
// length prefix combined with a few bytes of payload. Without the
// zero-progress guard, ReadUint{16,32,64}Slice recurses on c[buffered:]
// where buffered == 0, exhausting the goroutine stack.
type truncReader struct {
	buf       []byte
	advertise int // value returned by Size(); intentionally != len(buf)
	off       int
}

func newTruncReader(payload []byte, advertise int) *truncReader {
	return &truncReader{buf: payload, advertise: advertise}
}

func (r *truncReader) Read(p []byte) (int, error) {
	if r.off >= len(r.buf) {
		return 0, io.EOF
	}
	n := copy(p, r.buf[r.off:])
	r.off += n
	return n, nil
}

// Size reports the *advertised* size, NOT the actual remaining bytes.
// This is what triggered the original CVE: the lattigo path trusts
// r.Size() for capping its Peek length.
func (r *truncReader) Size() int {
	remain := r.advertise - r.off
	if remain < 0 {
		return 0
	}
	return remain
}

// Peek returns up to n bytes of the actual buffer, but does NOT return
// io.EOF when the request exceeds the actual buffer — instead it returns
// (short, nil). This is the exact misbehavior produced by the warp/pulsar
// regression seed; bufio.Reader can also reach this state when the
// underlying io.Reader returns (0, nil).
func (r *truncReader) Peek(n int) ([]byte, error) {
	avail := len(r.buf) - r.off
	if avail >= n {
		return r.buf[r.off : r.off+n], nil
	}
	// Short, non-erroring peek: the bug surface.
	return r.buf[r.off:], nil
}

func (r *truncReader) Discard(n int) (int, error) {
	avail := len(r.buf) - r.off
	if n > avail {
		r.off = len(r.buf)
		return avail, io.EOF
	}
	r.off += n
	return n, nil
}

// TestReadUint64Slice_ZeroProgress is the regression test for the DoS
// vulnerability discovered by github.com/luxfi/warp/pulsar's
// FuzzPulseDeserialize harness, regression seed ccdb090e0ca0007b. Prior
// to the fix in cap.go + the iterative rewrite of ReadUint{16,32,64}Slice,
// this test would crash with `runtime: goroutine stack exceeds limit`.
func TestReadUint64Slice_ZeroProgress(t *testing.T) {
	// Two bytes of payload, advertise 1 GiB. After decoding `buffered=0`
	// elements the buggy code recurses on c[0:] which equals c.
	r := newTruncReader([]byte{0x41, 0x42}, 1<<30)

	// Destination slice large enough to make the fault observable.
	c := make([]uint64, 1<<20)

	_, err := ReadUint64Slice(r, c)
	if !errors.Is(err, ErrZeroProgress) {
		t.Fatalf("expected ErrZeroProgress, got %v", err)
	}
}

// TestReadUint32Slice_ZeroProgress verifies the same guard for uint32.
func TestReadUint32Slice_ZeroProgress(t *testing.T) {
	r := newTruncReader([]byte{0x01}, 1<<30)
	c := make([]uint32, 1<<20)

	_, err := ReadUint32Slice(r, c)
	if !errors.Is(err, ErrZeroProgress) {
		t.Fatalf("expected ErrZeroProgress, got %v", err)
	}
}

// TestReadUint16Slice_ZeroProgress verifies the same guard for uint16.
func TestReadUint16Slice_ZeroProgress(t *testing.T) {
	r := newTruncReader([]byte{}, 1<<30)
	c := make([]uint16, 1<<20)

	_, err := ReadUint16Slice(r, c)
	if !errors.Is(err, ErrZeroProgress) {
		t.Fatalf("expected ErrZeroProgress, got %v", err)
	}
}

// TestReadUint64Slice_HappyPath ensures the iterative rewrite still
// decodes correctly when the Peek returns a partial slice that DOES
// permit progress (>= 8 bytes). This guards against regressing the
// original lattigo callers, which depend on multi-step decoding of
// large polynomials over a chunked Reader.
func TestReadUint64Slice_HappyPath(t *testing.T) {
	const N = 16
	payload := make([]byte, N*8)
	for i := 0; i < N; i++ {
		// Little-endian uint64 with i+1 in the low byte.
		payload[i*8] = byte(i + 1)
	}

	// Buffer-backed reader exercises the in-tree happy path.
	b := NewBuffer(payload)
	c := make([]uint64, N)

	n, err := ReadUint64Slice(b, c)
	if err != nil {
		t.Fatalf("ReadUint64Slice: %v", err)
	}
	if n != int64(N*8) {
		t.Fatalf("read %d bytes, want %d", n, N*8)
	}
	for i := 0; i < N; i++ {
		if c[i] != uint64(i+1) {
			t.Fatalf("c[%d] = %d, want %d", i, c[i], i+1)
		}
	}
}

// TestReadUint64SliceBounded_RejectsHugeLength asserts that the
// length-bounded wrapper rejects attacker-controlled slice lengths
// before any Peek is issued.
func TestReadUint64SliceBounded_RejectsHugeLength(t *testing.T) {
	r := NewBuffer([]byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08})
	_, _, err := ReadUint64SliceBounded(r, MaxSliceLen()+1)
	if !errors.Is(err, ErrSliceTooLarge) {
		t.Fatalf("expected ErrSliceTooLarge, got %v", err)
	}
	_, _, err = ReadUint64SliceBounded(r, -1)
	if !errors.Is(err, ErrSliceTooLarge) {
		t.Fatalf("expected ErrSliceTooLarge for negative, got %v", err)
	}
}

// TestSetMaxSliceLen confirms the cap can be tightened at runtime and
// reset to default with n <= 0.
func TestSetMaxSliceLen(t *testing.T) {
	t.Cleanup(func() { SetMaxSliceLen(0) })

	SetMaxSliceLen(64)
	if got := MaxSliceLen(); got != 64 {
		t.Fatalf("MaxSliceLen() = %d, want 64", got)
	}

	SetMaxSliceLen(0) // reset
	if got := MaxSliceLen(); got != DefaultMaxSliceLen {
		t.Fatalf("MaxSliceLen() = %d, want %d after reset", got, DefaultMaxSliceLen)
	}
}
