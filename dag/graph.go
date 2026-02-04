// Package dag provides a directed acyclic graph scheduler for FHE operations.
//
// DAG execution optimizes FHE operation scheduling by:
// - Building a computation graph from FHE operations
// - Identifying independent operations that can run in parallel
// - Minimizing GPU kernel launches via operation batching
// - Tracking ciphertext dependencies automatically
// - Managing memory for intermediate results
//
// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
package dag

import (
	"fmt"
	"sync/atomic"
)

// OpType represents the type of FHE operation.
type OpType uint8

const (
	// Arithmetic operations
	OpAdd OpType = iota
	OpSub
	OpMul
	OpMulPlain
	OpAddPlain
	OpSubPlain
	OpNegate

	// NTT operations
	OpNTT
	OpINTT

	// Key switching operations
	OpRelinearize
	OpRotate
	OpConjugate

	// Level management
	OpRescale
	OpModSwitch

	// Bootstrap
	OpBootstrap

	// Memory operations
	OpCopy
	OpLoad  // Load from external source
	OpStore // Store to external destination
)

// String returns the string representation of an OpType.
func (op OpType) String() string {
	names := []string{
		"Add", "Sub", "Mul", "MulPlain", "AddPlain", "SubPlain", "Negate",
		"NTT", "INTT",
		"Relinearize", "Rotate", "Conjugate",
		"Rescale", "ModSwitch",
		"Bootstrap",
		"Copy", "Load", "Store",
	}
	if int(op) < len(names) {
		return names[op]
	}
	return fmt.Sprintf("Unknown(%d)", op)
}

// CanFuse returns true if operations of this type can be fused.
func (op OpType) CanFuse() bool {
	switch op {
	case OpAdd, OpSub, OpMulPlain, OpAddPlain, OpSubPlain, OpNegate:
		return true
	default:
		return false
	}
}

// IsElementwise returns true if the operation is element-wise.
func (op OpType) IsElementwise() bool {
	switch op {
	case OpAdd, OpSub, OpMulPlain, OpAddPlain, OpSubPlain, OpNegate, OpNTT, OpINTT:
		return true
	default:
		return false
	}
}

// NodeID uniquely identifies a node in the DAG.
type NodeID uint64

// CiphertextID uniquely identifies a ciphertext in the computation.
type CiphertextID uint64

// nextNodeID is the global counter for generating unique node IDs.
var nextNodeID uint64

// newNodeID generates a new unique node ID.
func newNodeID() NodeID {
	return NodeID(atomic.AddUint64(&nextNodeID, 1))
}

// Node represents a single operation in the DAG.
type Node struct {
	ID       NodeID
	Op       OpType
	Inputs   []NodeID       // IDs of input nodes
	Output   CiphertextID   // ID of output ciphertext
	Level    int            // Ciphertext level after operation
	Metadata map[string]any // Operation-specific metadata

	// Scheduling information (set by scheduler)
	Depth    int  // Topological depth (0 = input nodes)
	BatchID  int  // Batch this node belongs to (-1 = not scheduled)
	Executed bool // Whether this node has been executed

	// Dependencies tracking
	dependsOn  []*Node // Nodes this node depends on
	dependents []*Node // Nodes that depend on this node
}

// NewNode creates a new DAG node with the given operation type.
func NewNode(op OpType) *Node {
	return &Node{
		ID:        newNodeID(),
		Op:        op,
		Inputs:    make([]NodeID, 0, 2),
		Metadata:  make(map[string]any),
		Depth:     -1,
		BatchID:   -1,
		dependsOn: make([]*Node, 0, 2),
	}
}

// AddInput adds an input node dependency.
func (n *Node) AddInput(inputID NodeID) {
	n.Inputs = append(n.Inputs, inputID)
}

// SetMetadata sets a metadata value for the node.
func (n *Node) SetMetadata(key string, value any) {
	n.Metadata[key] = value
}

// GetMetadata retrieves a metadata value.
func (n *Node) GetMetadata(key string) (any, bool) {
	v, ok := n.Metadata[key]
	return v, ok
}

// IsReady returns true if all input dependencies have been executed.
func (n *Node) IsReady() bool {
	for _, dep := range n.dependsOn {
		if !dep.Executed {
			return false
		}
	}
	return true
}

// Graph represents a DAG of FHE operations.
type Graph struct {
	nodes       map[NodeID]*Node
	outputs     map[CiphertextID]*Node // Map from ciphertext ID to producing node
	inputNodes  []*Node                // Nodes with no dependencies (inputs)
	outputNodes []*Node                // Nodes marked as outputs

	// Statistics
	nodeCount   int
	maxDepth    int
	parallelism float64 // Average operations per level
}

// NewGraph creates a new empty DAG.
func NewGraph() *Graph {
	return &Graph{
		nodes:       make(map[NodeID]*Node),
		outputs:     make(map[CiphertextID]*Node),
		inputNodes:  make([]*Node, 0),
		outputNodes: make([]*Node, 0),
	}
}

// AddNode adds a node to the graph.
func (g *Graph) AddNode(n *Node) error {
	if _, exists := g.nodes[n.ID]; exists {
		return fmt.Errorf("node %d already exists in graph", n.ID)
	}

	// Register the node
	g.nodes[n.ID] = n
	g.nodeCount++

	// Link dependencies
	for _, inputID := range n.Inputs {
		inputNode, ok := g.nodes[inputID]
		if !ok {
			return fmt.Errorf("input node %d not found for node %d", inputID, n.ID)
		}
		n.dependsOn = append(n.dependsOn, inputNode)
		inputNode.dependents = append(inputNode.dependents, n)
	}

	// Track input nodes (no dependencies)
	if len(n.Inputs) == 0 {
		g.inputNodes = append(g.inputNodes, n)
	}

	// Register output ciphertext
	if n.Output != 0 {
		g.outputs[n.Output] = n
	}

	return nil
}

// GetNode retrieves a node by ID.
func (g *Graph) GetNode(id NodeID) (*Node, bool) {
	n, ok := g.nodes[id]
	return n, ok
}

// GetProducer returns the node that produces a given ciphertext.
func (g *Graph) GetProducer(ctID CiphertextID) (*Node, bool) {
	n, ok := g.outputs[ctID]
	return n, ok
}

// MarkOutput marks a node as a graph output.
func (g *Graph) MarkOutput(id NodeID) error {
	n, ok := g.nodes[id]
	if !ok {
		return fmt.Errorf("node %d not found", id)
	}
	g.outputNodes = append(g.outputNodes, n)
	return nil
}

// NodeCount returns the number of nodes in the graph.
func (g *Graph) NodeCount() int {
	return g.nodeCount
}

// InputNodes returns nodes with no dependencies.
func (g *Graph) InputNodes() []*Node {
	return g.inputNodes
}

// OutputNodes returns nodes marked as outputs.
func (g *Graph) OutputNodes() []*Node {
	return g.outputNodes
}

// Nodes returns all nodes in the graph.
func (g *Graph) Nodes() []*Node {
	result := make([]*Node, 0, len(g.nodes))
	for _, n := range g.nodes {
		result = append(result, n)
	}
	return result
}

// MaxDepth returns the maximum depth of the DAG (set after scheduling).
func (g *Graph) MaxDepth() int {
	return g.maxDepth
}

// Parallelism returns the average number of operations per depth level.
func (g *Graph) Parallelism() float64 {
	return g.parallelism
}

// Validate checks the graph for consistency and cycles.
func (g *Graph) Validate() error {
	// Check for cycles using DFS
	visited := make(map[NodeID]bool)
	recStack := make(map[NodeID]bool)

	var hasCycle func(n *Node) bool
	hasCycle = func(n *Node) bool {
		visited[n.ID] = true
		recStack[n.ID] = true

		for _, dep := range n.dependents {
			if !visited[dep.ID] {
				if hasCycle(dep) {
					return true
				}
			} else if recStack[dep.ID] {
				return true
			}
		}

		recStack[n.ID] = false
		return false
	}

	for _, n := range g.nodes {
		if !visited[n.ID] {
			if hasCycle(n) {
				return fmt.Errorf("graph contains a cycle")
			}
		}
	}

	// Check all inputs are valid
	for _, n := range g.nodes {
		for _, inputID := range n.Inputs {
			if _, ok := g.nodes[inputID]; !ok {
				return fmt.Errorf("node %d references non-existent input %d", n.ID, inputID)
			}
		}
	}

	return nil
}

// Reset clears execution state for all nodes.
func (g *Graph) Reset() {
	for _, n := range g.nodes {
		n.Executed = false
		n.BatchID = -1
	}
}

// Builder provides a fluent API for constructing DAGs.
type Builder struct {
	graph    *Graph
	nextCtID CiphertextID
	lastNode *Node
	err      error
}

// NewBuilder creates a new DAG builder.
func NewBuilder() *Builder {
	return &Builder{
		graph:    NewGraph(),
		nextCtID: 1,
	}
}

// Input adds an input node representing an external ciphertext.
func (b *Builder) Input(level int) *Builder {
	if b.err != nil {
		return b
	}

	n := NewNode(OpLoad)
	n.Output = b.nextCtID
	b.nextCtID++
	n.Level = level

	if err := b.graph.AddNode(n); err != nil {
		b.err = err
		return b
	}

	b.lastNode = n
	return b
}

// Add adds an addition operation.
func (b *Builder) Add(input1, input2 NodeID) *Builder {
	return b.binaryOp(OpAdd, input1, input2)
}

// Sub adds a subtraction operation.
func (b *Builder) Sub(input1, input2 NodeID) *Builder {
	return b.binaryOp(OpSub, input1, input2)
}

// Mul adds a multiplication operation.
func (b *Builder) Mul(input1, input2 NodeID) *Builder {
	return b.binaryOp(OpMul, input1, input2)
}

// Relin adds a relinearization operation.
func (b *Builder) Relin(input NodeID) *Builder {
	return b.unaryOp(OpRelinearize, input)
}

// Rescale adds a rescaling operation.
func (b *Builder) Rescale(input NodeID) *Builder {
	return b.unaryOp(OpRescale, input)
}

// Rotate adds a rotation operation.
func (b *Builder) Rotate(input NodeID, rotIdx int) *Builder {
	b.unaryOp(OpRotate, input)
	if b.err == nil {
		b.lastNode.SetMetadata("rotation", rotIdx)
	}
	return b
}

// NTT adds an NTT operation.
func (b *Builder) NTT(input NodeID) *Builder {
	return b.unaryOp(OpNTT, input)
}

// INTT adds an inverse NTT operation.
func (b *Builder) INTT(input NodeID) *Builder {
	return b.unaryOp(OpINTT, input)
}

// Output marks the last added node as an output.
func (b *Builder) Output() *Builder {
	if b.err != nil || b.lastNode == nil {
		return b
	}
	if err := b.graph.MarkOutput(b.lastNode.ID); err != nil {
		b.err = err
	}
	return b
}

// LastNodeID returns the ID of the last added node.
func (b *Builder) LastNodeID() NodeID {
	if b.lastNode == nil {
		return 0
	}
	return b.lastNode.ID
}

// Build finalizes and returns the graph.
func (b *Builder) Build() (*Graph, error) {
	if b.err != nil {
		return nil, b.err
	}
	if err := b.graph.Validate(); err != nil {
		return nil, err
	}
	return b.graph, nil
}

// binaryOp adds a binary operation node.
func (b *Builder) binaryOp(op OpType, input1, input2 NodeID) *Builder {
	if b.err != nil {
		return b
	}

	n := NewNode(op)
	n.AddInput(input1)
	n.AddInput(input2)
	n.Output = b.nextCtID
	b.nextCtID++

	// Determine output level from inputs
	n1, ok1 := b.graph.GetNode(input1)
	n2, ok2 := b.graph.GetNode(input2)
	if ok1 && ok2 {
		n.Level = min(n1.Level, n2.Level)
	}

	if err := b.graph.AddNode(n); err != nil {
		b.err = err
		return b
	}

	b.lastNode = n
	return b
}

// unaryOp adds a unary operation node.
func (b *Builder) unaryOp(op OpType, input NodeID) *Builder {
	if b.err != nil {
		return b
	}

	n := NewNode(op)
	n.AddInput(input)
	n.Output = b.nextCtID
	b.nextCtID++

	// Inherit level from input
	if inputNode, ok := b.graph.GetNode(input); ok {
		n.Level = inputNode.Level
		if op == OpRescale {
			n.Level-- // Rescale reduces level by 1
		}
	}

	if err := b.graph.AddNode(n); err != nil {
		b.err = err
		return b
	}

	b.lastNode = n
	return b
}
