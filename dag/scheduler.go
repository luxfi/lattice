// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause

package dag

import (
	"fmt"
	"sort"
)

// Batch represents a group of independent operations that can execute in parallel.
type Batch struct {
	ID     int
	Depth  int
	Nodes  []*Node
	OpType OpType // Primary operation type (for homogeneous batches)
	Mixed  bool   // True if batch contains mixed operation types
}

// Size returns the number of nodes in the batch.
func (b *Batch) Size() int {
	return len(b.Nodes)
}

// Schedule represents a complete execution schedule for a DAG.
type Schedule struct {
	Batches      []*Batch
	NodeToBatch  map[NodeID]int
	TotalNodes   int
	MaxParallel  int     // Maximum batch size
	AvgParallel  float64 // Average batch size
	CriticalPath int     // Length of critical path (depth)
}

// Scheduler handles DAG analysis and batch scheduling.
type Scheduler struct {
	graph    *Graph
	schedule *Schedule

	// Configuration
	maxBatchSize    int  // Maximum operations per batch
	fuseSimilarOps  bool // Fuse similar operations in same batch
	minimizeKernels bool // Prioritize fewer kernel launches
}

// SchedulerOption configures the scheduler.
type SchedulerOption func(*Scheduler)

// WithMaxBatchSize sets the maximum batch size.
func WithMaxBatchSize(size int) SchedulerOption {
	return func(s *Scheduler) {
		s.maxBatchSize = size
	}
}

// WithFuseSimilarOps enables/disables operation fusion.
func WithFuseSimilarOps(enable bool) SchedulerOption {
	return func(s *Scheduler) {
		s.fuseSimilarOps = enable
	}
}

// WithMinimizeKernels enables kernel minimization mode.
func WithMinimizeKernels(enable bool) SchedulerOption {
	return func(s *Scheduler) {
		s.minimizeKernels = enable
	}
}

// NewScheduler creates a new scheduler for the given graph.
func NewScheduler(g *Graph, opts ...SchedulerOption) *Scheduler {
	s := &Scheduler{
		graph:           g,
		maxBatchSize:    1024, // Default max batch size
		fuseSimilarOps:  true,
		minimizeKernels: true,
	}

	for _, opt := range opts {
		opt(s)
	}

	return s
}

// Schedule computes an execution schedule for the DAG.
func (s *Scheduler) Schedule() (*Schedule, error) {
	if err := s.graph.Validate(); err != nil {
		return nil, fmt.Errorf("invalid graph: %w", err)
	}

	// Step 1: Compute topological depths
	if err := s.computeDepths(); err != nil {
		return nil, fmt.Errorf("failed to compute depths: %w", err)
	}

	// Step 2: Group nodes by depth level
	levelNodes := s.groupByLevel()

	// Step 3: Create batches from level groups
	batches := s.createBatches(levelNodes)

	// Step 4: Build schedule
	s.schedule = &Schedule{
		Batches:      batches,
		NodeToBatch:  make(map[NodeID]int),
		TotalNodes:   s.graph.NodeCount(),
		CriticalPath: s.graph.maxDepth,
	}

	// Map nodes to batches and compute statistics
	maxParallel := 0
	totalParallel := 0
	for _, batch := range batches {
		for _, node := range batch.Nodes {
			s.schedule.NodeToBatch[node.ID] = batch.ID
			node.BatchID = batch.ID
		}
		if batch.Size() > maxParallel {
			maxParallel = batch.Size()
		}
		totalParallel += batch.Size()
	}

	s.schedule.MaxParallel = maxParallel
	if len(batches) > 0 {
		s.schedule.AvgParallel = float64(totalParallel) / float64(len(batches))
	}

	return s.schedule, nil
}

// computeDepths assigns topological depths to all nodes.
// Depth is the longest path from any input to the node.
func (s *Scheduler) computeDepths() error {
	// Initialize all depths to -1
	for _, n := range s.graph.nodes {
		n.Depth = -1
	}

	// Set input nodes to depth 0
	for _, n := range s.graph.inputNodes {
		n.Depth = 0
	}

	// BFS to propagate depths
	queue := make([]*Node, len(s.graph.inputNodes))
	copy(queue, s.graph.inputNodes)

	maxDepth := 0
	processed := 0

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		processed++

		// Update depths of dependents
		for _, dep := range current.dependents {
			newDepth := current.Depth + 1
			if newDepth > dep.Depth {
				dep.Depth = newDepth
				if newDepth > maxDepth {
					maxDepth = newDepth
				}
			}

			// Check if all dependencies are processed
			allDepsProcessed := true
			for _, d := range dep.dependsOn {
				if d.Depth == -1 {
					allDepsProcessed = false
					break
				}
			}

			// Add to queue if all dependencies have depths
			if allDepsProcessed && dep.Depth != -1 {
				// Avoid duplicates in queue
				inQueue := false
				for _, q := range queue {
					if q.ID == dep.ID {
						inQueue = true
						break
					}
				}
				if !inQueue {
					queue = append(queue, dep)
				}
			}
		}
	}

	// Verify all nodes have been assigned depths
	for _, n := range s.graph.nodes {
		if n.Depth == -1 {
			return fmt.Errorf("node %d unreachable from inputs", n.ID)
		}
	}

	s.graph.maxDepth = maxDepth

	// Compute parallelism
	if maxDepth > 0 {
		s.graph.parallelism = float64(s.graph.nodeCount) / float64(maxDepth+1)
	}

	return nil
}

// groupByLevel groups nodes by their topological depth.
func (s *Scheduler) groupByLevel() [][]*Node {
	maxDepth := s.graph.maxDepth
	levels := make([][]*Node, maxDepth+1)

	for i := range levels {
		levels[i] = make([]*Node, 0)
	}

	for _, n := range s.graph.nodes {
		levels[n.Depth] = append(levels[n.Depth], n)
	}

	return levels
}

// createBatches creates batches from level groups.
func (s *Scheduler) createBatches(levelNodes [][]*Node) []*Batch {
	batches := make([]*Batch, 0)
	batchID := 0

	for depth, nodes := range levelNodes {
		if len(nodes) == 0 {
			continue
		}

		if s.fuseSimilarOps {
			// Group by operation type within level
			opGroups := s.groupByOpType(nodes)

			for opType, opNodes := range opGroups {
				// Split into batches respecting maxBatchSize
				for i := 0; i < len(opNodes); i += s.maxBatchSize {
					end := i + s.maxBatchSize
					if end > len(opNodes) {
						end = len(opNodes)
					}

					batch := &Batch{
						ID:     batchID,
						Depth:  depth,
						Nodes:  opNodes[i:end],
						OpType: opType,
						Mixed:  false,
					}
					batches = append(batches, batch)
					batchID++
				}
			}
		} else {
			// Create mixed batches
			for i := 0; i < len(nodes); i += s.maxBatchSize {
				end := i + s.maxBatchSize
				if end > len(nodes) {
					end = len(nodes)
				}

				batch := &Batch{
					ID:    batchID,
					Depth: depth,
					Nodes: nodes[i:end],
					Mixed: len(s.groupByOpType(nodes[i:end])) > 1,
				}
				if !batch.Mixed && len(batch.Nodes) > 0 {
					batch.OpType = batch.Nodes[0].Op
				}
				batches = append(batches, batch)
				batchID++
			}
		}
	}

	return batches
}

// groupByOpType groups nodes by their operation type.
func (s *Scheduler) groupByOpType(nodes []*Node) map[OpType][]*Node {
	groups := make(map[OpType][]*Node)

	for _, n := range nodes {
		groups[n.Op] = append(groups[n.Op], n)
	}

	return groups
}

// GetSchedule returns the computed schedule.
func (s *Scheduler) GetSchedule() *Schedule {
	return s.schedule
}

// ScheduleStats provides statistics about a schedule.
type ScheduleStats struct {
	TotalBatches     int
	TotalNodes       int
	MaxBatchSize     int
	MinBatchSize     int
	AvgBatchSize     float64
	CriticalPath     int
	Parallelism      float64 // Nodes / Critical path length
	KernelEfficiency float64 // Nodes per kernel launch estimate
}

// Stats computes statistics for the schedule.
func (s *Schedule) Stats() ScheduleStats {
	stats := ScheduleStats{
		TotalBatches: len(s.Batches),
		TotalNodes:   s.TotalNodes,
		MaxBatchSize: s.MaxParallel,
		CriticalPath: s.CriticalPath,
	}

	if len(s.Batches) == 0 {
		return stats
	}

	stats.MinBatchSize = s.Batches[0].Size()
	totalSize := 0

	for _, batch := range s.Batches {
		size := batch.Size()
		totalSize += size
		if size < stats.MinBatchSize {
			stats.MinBatchSize = size
		}
	}

	stats.AvgBatchSize = float64(totalSize) / float64(len(s.Batches))

	if stats.CriticalPath > 0 {
		stats.Parallelism = float64(stats.TotalNodes) / float64(stats.CriticalPath)
	}

	// Estimate kernel efficiency based on batch homogeneity
	homogeneousBatches := 0
	for _, batch := range s.Batches {
		if !batch.Mixed {
			homogeneousBatches++
		}
	}
	if len(s.Batches) > 0 {
		stats.KernelEfficiency = float64(stats.TotalNodes) / float64(len(s.Batches))
	}

	return stats
}

// MemoryEstimate estimates peak memory usage in ciphertext count.
type MemoryEstimate struct {
	PeakCiphertexts    int // Maximum concurrent ciphertexts
	TotalIntermediates int // Total intermediate ciphertexts created
	ReusableBufs       int // Buffers that can be reused
}

// EstimateMemory estimates memory requirements for the schedule.
func (s *Schedule) EstimateMemory() MemoryEstimate {
	est := MemoryEstimate{}

	// Track live ciphertexts at each batch
	live := make(map[CiphertextID]bool)
	peak := 0

	for _, batch := range s.Batches {
		// Add outputs
		for _, node := range batch.Nodes {
			if node.Output != 0 {
				live[node.Output] = true
				est.TotalIntermediates++
			}
		}

		// Update peak
		if len(live) > peak {
			peak = len(live)
		}

		// Note: A full implementation would track when ciphertexts
		// are no longer needed and mark them for reuse
	}

	est.PeakCiphertexts = peak
	return est
}

// TopologicalOrder returns nodes in topological order.
func TopologicalOrder(g *Graph) ([]*Node, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}

	// Kahn's algorithm
	inDegree := make(map[NodeID]int)
	for _, n := range g.nodes {
		inDegree[n.ID] = len(n.dependsOn)
	}

	// Start with nodes that have no dependencies
	queue := make([]*Node, 0)
	for _, n := range g.nodes {
		if inDegree[n.ID] == 0 {
			queue = append(queue, n)
		}
	}

	// Sort initial queue by ID for deterministic output
	sort.Slice(queue, func(i, j int) bool {
		return queue[i].ID < queue[j].ID
	})

	result := make([]*Node, 0, len(g.nodes))

	for len(queue) > 0 {
		// Pop front
		n := queue[0]
		queue = queue[1:]
		result = append(result, n)

		// Reduce in-degree of dependents
		for _, dep := range n.dependents {
			inDegree[dep.ID]--
			if inDegree[dep.ID] == 0 {
				queue = append(queue, dep)
			}
		}

		// Sort for determinism
		sort.Slice(queue, func(i, j int) bool {
			return queue[i].ID < queue[j].ID
		})
	}

	if len(result) != len(g.nodes) {
		return nil, fmt.Errorf("cycle detected: processed %d of %d nodes", len(result), len(g.nodes))
	}

	return result, nil
}

// CriticalPath returns the nodes on the critical path (longest dependency chain).
func CriticalPath(g *Graph) ([]*Node, error) {
	ordered, err := TopologicalOrder(g)
	if err != nil {
		return nil, err
	}

	// Compute longest path to each node
	dist := make(map[NodeID]int)
	parent := make(map[NodeID]*Node)

	for _, n := range ordered {
		maxDist := 0
		var maxParent *Node
		for _, dep := range n.dependsOn {
			if dist[dep.ID]+1 > maxDist {
				maxDist = dist[dep.ID] + 1
				maxParent = dep
			}
		}
		dist[n.ID] = maxDist
		parent[n.ID] = maxParent
	}

	// Find the node with maximum distance
	var endNode *Node
	maxDist := -1
	for _, n := range g.nodes {
		if dist[n.ID] > maxDist {
			maxDist = dist[n.ID]
			endNode = n
		}
	}

	if endNode == nil {
		return nil, nil
	}

	// Trace back the critical path
	path := make([]*Node, 0)
	current := endNode
	for current != nil {
		path = append([]*Node{current}, path...)
		current = parent[current.ID]
	}

	return path, nil
}
