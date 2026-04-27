package blindrot

import (
	"errors"
	"runtime"
	"sync"

	"github.com/luxfi/lattice/v7/core/rlwe"
	"github.com/luxfi/lattice/v7/ring"
)

// errMismatchedTestPolyLen is returned for every input slot when len(testPolys)
// does not match len(cts). The whole batch is rejected — failing per-slot here
// would mask a programmer error.
var errMismatchedTestPolyLen = errors.New("blindrot: BatchEvaluate: len(testPolys) != len(cts)")

// BatchEvaluate runs N independent blind-rotation evaluations in parallel.
//
// Each entry in cts is paired with the corresponding entry in testPolys; the
// per-iteration call is byte-equivalent to a serial Evaluate(ct, tp, BRK).
//
// Concurrency model: the receiver Evaluator carries mutable scratch buffers
// (poolMod2N, accumulator) and an embedded *rgsw.Evaluator with its own buffer
// pool. Those structures are NOT safe for concurrent use, and the package
// exposes no ShallowCopy. BatchEvaluate therefore allocates one fresh
// Evaluator per worker goroutine via NewEvaluator(paramsBR, paramsLWE) and
// fans the work out across runtime.GOMAXPROCS workers. The receiver Evaluator
// is read-only here (only paramsBR and paramsLWE are inspected for cloning).
//
// The BlindRotationEvaluationKeySet is treated as read-only and shared across
// goroutines: GetBlindRotationKey / GetEvaluationKeySet must return the same
// keys for the same indices (the in-tree MemBlindRotationEvaluationKeySet
// satisfies this — it is a fixed slice of pointers).
//
// Per PERFORMANCE.md §4 (luxfi/fhe), the M1 Max P-core ceiling is ~5.84× at
// N≥16; the goroutine fan-out saturates the 8 P-cores. The Metal batch path
// is excluded by design until the metal_batch_bootstrap kernel ships in
// luxcpp/lattice/src/metal/ (PERFORMANCE.md §6 G3). When it lands, this
// function gains a fast path that dispatches the whole batch to one Metal
// command buffer instead of fanning out goroutines.
func (eval *Evaluator) BatchEvaluate(
	cts []*rlwe.Ciphertext,
	testPolys []map[int]*ring.Poly,
	BRK BlindRotationEvaluationKeySet,
) ([]map[int]*rlwe.Ciphertext, []error) {

	n := len(cts)
	if n == 0 {
		return nil, nil
	}
	if len(testPolys) != n {
		errs := make([]error, n)
		for i := range errs {
			errs[i] = errMismatchedTestPolyLen
		}
		return make([]map[int]*rlwe.Ciphertext, n), errs
	}

	out := make([]map[int]*rlwe.Ciphertext, n)
	errs := make([]error, n)

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}

	// Single-worker fast path — fall through to a plain serial loop on the
	// receiver. Avoids the cost of spinning up a worker pool for n=1.
	if workers == 1 {
		for i := range cts {
			out[i], errs[i] = eval.Evaluate(cts[i], testPolys[i], BRK)
		}
		return out, errs
	}

	// Fan-out: each worker owns one fresh Evaluator. Re-creating an Evaluator
	// is cheap relative to a single Evaluate call (one ring poly pair + one
	// ciphertext + the discrete-log map; ~N entries). The BRK is shared.
	jobs := make(chan int, n)
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)

	paramsBR := eval.paramsBR
	paramsLWE := eval.paramsLWE

	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			worker := NewEvaluator(paramsBR, paramsLWE)
			for i := range jobs {
				out[i], errs[i] = worker.Evaluate(cts[i], testPolys[i], BRK)
			}
		}()
	}
	wg.Wait()
	return out, errs
}
