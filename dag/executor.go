// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause

package dag

import (
	"context"
	"fmt"
	"sync"

	"github.com/luxfi/lattice/v7/ring"
)

// CiphertextPool manages a pool of ciphertext buffers for reuse.
type CiphertextPool struct {
	mu       sync.Mutex
	buffers  [][]ring.Poly
	ringQ    *ring.Ring
	degree   int
	maxLevel int
}

// NewCiphertextPool creates a new ciphertext buffer pool.
func NewCiphertextPool(ringQ *ring.Ring, degree, maxLevel, initialSize int) *CiphertextPool {
	pool := &CiphertextPool{
		ringQ:    ringQ,
		degree:   degree,
		maxLevel: maxLevel,
		buffers:  make([][]ring.Poly, 0, initialSize),
	}

	// Pre-allocate buffers
	for i := 0; i < initialSize; i++ {
		pool.buffers = append(pool.buffers, pool.allocate())
	}

	return pool
}

// Get retrieves a buffer from the pool or allocates a new one.
func (p *CiphertextPool) Get() []ring.Poly {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.buffers) == 0 {
		return p.allocate()
	}

	buf := p.buffers[len(p.buffers)-1]
	p.buffers = p.buffers[:len(p.buffers)-1]
	return buf
}

// Put returns a buffer to the pool.
func (p *CiphertextPool) Put(buf []ring.Poly) {
	if buf == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.buffers = append(p.buffers, buf)
}

// allocate creates a new ciphertext buffer.
func (p *CiphertextPool) allocate() []ring.Poly {
	polys := make([]ring.Poly, p.degree+1)
	for i := range polys {
		polys[i] = p.ringQ.NewPoly()
	}
	return polys
}

// Clear releases all pooled buffers.
func (p *CiphertextPool) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.buffers = p.buffers[:0]
}

// ExecutionContext holds the state for DAG execution.
type ExecutionContext struct {
	// Ciphertext storage indexed by CiphertextID
	ciphertexts sync.Map // map[CiphertextID][]ring.Poly

	// Buffer pool for intermediates
	pool *CiphertextPool

	// GPU context (if available)
	gpuCtx any // Will be *gpu.NTTContext when GPU is available

	// Ring parameters
	ringQ  *ring.Ring
	degree int
	maxLevel int

	// Statistics
	opsExecuted   int64
	batchesRun    int64
	gpuDispatches int64
}

// NewExecutionContext creates a new execution context.
func NewExecutionContext(ringQ *ring.Ring, degree, maxLevel int) *ExecutionContext {
	poolSize := 32 // Initial pool size
	return &ExecutionContext{
		ringQ:    ringQ,
		degree:   degree,
		maxLevel: maxLevel,
		pool:     NewCiphertextPool(ringQ, degree, maxLevel, poolSize),
	}
}

// SetGPUContext sets the GPU context for accelerated execution.
func (ec *ExecutionContext) SetGPUContext(ctx any) {
	ec.gpuCtx = ctx
}

// SetCiphertext stores a ciphertext in the context.
func (ec *ExecutionContext) SetCiphertext(id CiphertextID, polys []ring.Poly) {
	ec.ciphertexts.Store(id, polys)
}

// GetCiphertext retrieves a ciphertext from the context.
func (ec *ExecutionContext) GetCiphertext(id CiphertextID) ([]ring.Poly, bool) {
	v, ok := ec.ciphertexts.Load(id)
	if !ok {
		return nil, false
	}
	return v.([]ring.Poly), true
}

// ReleaseCiphertext releases a ciphertext back to the pool.
func (ec *ExecutionContext) ReleaseCiphertext(id CiphertextID) {
	v, ok := ec.ciphertexts.LoadAndDelete(id)
	if ok {
		ec.pool.Put(v.([]ring.Poly))
	}
}

// AllocateCiphertext allocates a new ciphertext buffer.
func (ec *ExecutionContext) AllocateCiphertext() []ring.Poly {
	return ec.pool.Get()
}

// Stats returns execution statistics.
func (ec *ExecutionContext) Stats() (ops, batches, gpuDispatches int64) {
	return ec.opsExecuted, ec.batchesRun, ec.gpuDispatches
}

// OperationFunc is a function that executes a single FHE operation.
type OperationFunc func(ctx *ExecutionContext, node *Node, level int) error

// Executor executes DAG schedules.
type Executor struct {
	schedule *Schedule
	ctx      *ExecutionContext

	// Operation implementations
	ops map[OpType]OperationFunc

	// Configuration
	enableGPU   bool
	parallelCPU int // Number of parallel CPU workers
}

// ExecutorOption configures the executor.
type ExecutorOption func(*Executor)

// WithGPU enables/disables GPU execution.
func WithGPU(enable bool) ExecutorOption {
	return func(e *Executor) {
		e.enableGPU = enable
	}
}

// WithParallelCPU sets the number of parallel CPU workers.
func WithParallelCPU(workers int) ExecutorOption {
	return func(e *Executor) {
		e.parallelCPU = workers
	}
}

// NewExecutor creates a new executor for a schedule.
func NewExecutor(schedule *Schedule, ctx *ExecutionContext, opts ...ExecutorOption) *Executor {
	e := &Executor{
		schedule:    schedule,
		ctx:         ctx,
		ops:         make(map[OpType]OperationFunc),
		enableGPU:   true,
		parallelCPU: 4,
	}

	for _, opt := range opts {
		opt(e)
	}

	// Register default operations
	e.registerDefaultOps()

	return e
}

// RegisterOp registers an operation implementation.
func (e *Executor) RegisterOp(op OpType, fn OperationFunc) {
	e.ops[op] = fn
}

// registerDefaultOps registers default operation implementations.
func (e *Executor) registerDefaultOps() {
	// Add operation
	e.ops[OpAdd] = func(ctx *ExecutionContext, node *Node, level int) error {
		if len(node.Inputs) != 2 {
			return fmt.Errorf("Add requires 2 inputs, got %d", len(node.Inputs))
		}

		in1, ok1 := ctx.GetCiphertext(CiphertextID(node.Inputs[0]))
		in2, ok2 := ctx.GetCiphertext(CiphertextID(node.Inputs[1]))
		if !ok1 || !ok2 {
			return fmt.Errorf("input ciphertexts not found")
		}

		out := ctx.AllocateCiphertext()

		// Element-wise addition
		for i := range out {
			if i < len(in1) && i < len(in2) {
				ctx.ringQ.Add(in1[i], in2[i], out[i])
			}
		}

		ctx.SetCiphertext(node.Output, out)
		return nil
	}

	// Sub operation
	e.ops[OpSub] = func(ctx *ExecutionContext, node *Node, level int) error {
		if len(node.Inputs) != 2 {
			return fmt.Errorf("Sub requires 2 inputs, got %d", len(node.Inputs))
		}

		in1, ok1 := ctx.GetCiphertext(CiphertextID(node.Inputs[0]))
		in2, ok2 := ctx.GetCiphertext(CiphertextID(node.Inputs[1]))
		if !ok1 || !ok2 {
			return fmt.Errorf("input ciphertexts not found")
		}

		out := ctx.AllocateCiphertext()

		for i := range out {
			if i < len(in1) && i < len(in2) {
				ctx.ringQ.Sub(in1[i], in2[i], out[i])
			}
		}

		ctx.SetCiphertext(node.Output, out)
		return nil
	}

	// Negate operation
	e.ops[OpNegate] = func(ctx *ExecutionContext, node *Node, level int) error {
		if len(node.Inputs) != 1 {
			return fmt.Errorf("negate requires 1 input, got %d", len(node.Inputs))
		}

		in, ok := ctx.GetCiphertext(CiphertextID(node.Inputs[0]))
		if !ok {
			return fmt.Errorf("input ciphertext not found")
		}

		out := ctx.AllocateCiphertext()

		for i := range out {
			if i < len(in) {
				ctx.ringQ.Neg(in[i], out[i])
			}
		}

		ctx.SetCiphertext(node.Output, out)
		return nil
	}

	// NTT operation
	e.ops[OpNTT] = func(ctx *ExecutionContext, node *Node, level int) error {
		if len(node.Inputs) != 1 {
			return fmt.Errorf("NTT requires 1 input, got %d", len(node.Inputs))
		}

		in, ok := ctx.GetCiphertext(CiphertextID(node.Inputs[0]))
		if !ok {
			return fmt.Errorf("input ciphertext not found")
		}

		out := ctx.AllocateCiphertext()

		for i := range out {
			if i < len(in) {
				ctx.ringQ.NTT(in[i], out[i])
			}
		}

		ctx.SetCiphertext(node.Output, out)
		return nil
	}

	// INTT operation
	e.ops[OpINTT] = func(ctx *ExecutionContext, node *Node, level int) error {
		if len(node.Inputs) != 1 {
			return fmt.Errorf("INTT requires 1 input, got %d", len(node.Inputs))
		}

		in, ok := ctx.GetCiphertext(CiphertextID(node.Inputs[0]))
		if !ok {
			return fmt.Errorf("input ciphertext not found")
		}

		out := ctx.AllocateCiphertext()

		for i := range out {
			if i < len(in) {
				ctx.ringQ.INTT(in[i], out[i])
			}
		}

		ctx.SetCiphertext(node.Output, out)
		return nil
	}

	// Copy operation
	e.ops[OpCopy] = func(ctx *ExecutionContext, node *Node, level int) error {
		if len(node.Inputs) != 1 {
			return fmt.Errorf("copy requires 1 input, got %d", len(node.Inputs))
		}

		in, ok := ctx.GetCiphertext(CiphertextID(node.Inputs[0]))
		if !ok {
			return fmt.Errorf("input ciphertext not found")
		}

		out := ctx.AllocateCiphertext()

		for i := range out {
			if i < len(in) {
				copy(out[i].Coeffs[0], in[i].Coeffs[0])
			}
		}

		ctx.SetCiphertext(node.Output, out)
		return nil
	}

	// Load operation (no-op, assumes ciphertext is already set)
	e.ops[OpLoad] = func(ctx *ExecutionContext, node *Node, level int) error {
		// Load expects the ciphertext to be pre-set in context
		if _, ok := ctx.GetCiphertext(node.Output); !ok {
			return fmt.Errorf("load: ciphertext %d not found in context", node.Output)
		}
		return nil
	}

	// Store operation (no-op, just marks for output)
	e.ops[OpStore] = func(ctx *ExecutionContext, node *Node, level int) error {
		return nil
	}
}

// Execute runs the entire schedule.
func (e *Executor) Execute(ctx context.Context) error {
	for _, batch := range e.schedule.Batches {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err := e.executeBatch(batch); err != nil {
			return fmt.Errorf("batch %d failed: %w", batch.ID, err)
		}
		e.ctx.batchesRun++
	}
	return nil
}

// executeBatch executes a single batch.
func (e *Executor) executeBatch(batch *Batch) error {
	if len(batch.Nodes) == 0 {
		return nil
	}

	// Check if we can use GPU batched execution
	if e.enableGPU && !batch.Mixed && e.canBatchOnGPU(batch.OpType) {
		return e.executeGPUBatch(batch)
	}

	// Fall back to CPU execution
	return e.executeCPUBatch(batch)
}

// canBatchOnGPU checks if an operation type can be batched on GPU.
func (e *Executor) canBatchOnGPU(op OpType) bool {
	switch op {
	case OpAdd, OpSub, OpNTT, OpINTT, OpNegate:
		return e.ctx.gpuCtx != nil
	default:
		return false
	}
}

// executeGPUBatch executes a homogeneous batch on GPU.
func (e *Executor) executeGPUBatch(batch *Batch) error {
	// For now, dispatch individual operations but track as single GPU dispatch
	// A full implementation would pack data and call batched GPU kernels
	e.ctx.gpuDispatches++

	for _, node := range batch.Nodes {
		opFn, ok := e.ops[node.Op]
		if !ok {
			return fmt.Errorf("no implementation for operation %s", node.Op)
		}

		if err := opFn(e.ctx, node, node.Level); err != nil {
			return fmt.Errorf("node %d (%s) failed: %w", node.ID, node.Op, err)
		}

		node.Executed = true
		e.ctx.opsExecuted++
	}

	return nil
}

// executeCPUBatch executes a batch on CPU.
func (e *Executor) executeCPUBatch(batch *Batch) error {
	// For small batches, execute sequentially
	if len(batch.Nodes) <= e.parallelCPU {
		for _, node := range batch.Nodes {
			opFn, ok := e.ops[node.Op]
			if !ok {
				return fmt.Errorf("no implementation for operation %s", node.Op)
			}

			if err := opFn(e.ctx, node, node.Level); err != nil {
				return fmt.Errorf("node %d (%s) failed: %w", node.ID, node.Op, err)
			}

			node.Executed = true
			e.ctx.opsExecuted++
		}
		return nil
	}

	// For larger batches, execute in parallel
	var wg sync.WaitGroup
	errCh := make(chan error, len(batch.Nodes))

	for _, node := range batch.Nodes {
		wg.Add(1)
		go func(n *Node) {
			defer wg.Done()

			opFn, ok := e.ops[n.Op]
			if !ok {
				errCh <- fmt.Errorf("no implementation for operation %s", n.Op)
				return
			}

			if err := opFn(e.ctx, n, n.Level); err != nil {
				errCh <- fmt.Errorf("node %d (%s) failed: %w", n.ID, n.Op, err)
				return
			}

			n.Executed = true
			e.ctx.opsExecuted++
		}(node)
	}

	wg.Wait()
	close(errCh)

	// Return first error if any
	for err := range errCh {
		if err != nil {
			return err
		}
	}

	return nil
}

// BatchedGPUDispatch holds data for a GPU kernel dispatch.
type BatchedGPUDispatch struct {
	OpType    OpType
	InputIDs  [][]CiphertextID // Input ciphertext IDs per operation
	OutputIDs []CiphertextID   // Output ciphertext IDs
	Metadata  []map[string]any // Per-operation metadata
	BatchSize int
}

// PrepareBatchDispatch prepares a batch for GPU dispatch.
func PrepareBatchDispatch(batch *Batch) *BatchedGPUDispatch {
	dispatch := &BatchedGPUDispatch{
		OpType:    batch.OpType,
		InputIDs:  make([][]CiphertextID, len(batch.Nodes)),
		OutputIDs: make([]CiphertextID, len(batch.Nodes)),
		Metadata:  make([]map[string]any, len(batch.Nodes)),
		BatchSize: len(batch.Nodes),
	}

	for i, node := range batch.Nodes {
		dispatch.InputIDs[i] = make([]CiphertextID, len(node.Inputs))
		for j, input := range node.Inputs {
			dispatch.InputIDs[i][j] = CiphertextID(input)
		}
		dispatch.OutputIDs[i] = node.Output
		dispatch.Metadata[i] = node.Metadata
	}

	return dispatch
}

// ExecuteDAG is a convenience function to execute a DAG.
func ExecuteDAG(ctx context.Context, graph *Graph, execCtx *ExecutionContext) error {
	// Create scheduler and schedule
	scheduler := NewScheduler(graph)
	schedule, err := scheduler.Schedule()
	if err != nil {
		return fmt.Errorf("scheduling failed: %w", err)
	}

	// Create executor and run
	executor := NewExecutor(schedule, execCtx)
	return executor.Execute(ctx)
}
