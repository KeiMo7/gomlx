// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package memory

import (
	"slices"

	"github.com/gomlx/gomlx/internal/graphimpl"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/pkg/errors"
)

// RematerializationResult reports what an automatic rematerialization pass did.
type RematerializationResult struct {
	// Boundaries lists the node ids chosen as new checkpoint boundaries.
	Boundaries []graphimpl.NodeId

	// Plan is the snapshot computed after the boundaries were inserted.
	Plan *Plan

	// Applied is false when no boundary could reduce the peak further.
	Applied bool
}

// nonRematerializableTypes lists operators whose reconstruction is not safe to
// schedule lazily during a backward sweep (stateful, ordering-sensitive or
// control-flow nodes). A chain containing one of these cannot be automatically
// rematerialized across that node.
var nonRematerializableTypes = map[graphimpl.NodeType]bool{
	graphimpl.NodeTypeRNGBitGenerator:     true,
	graphimpl.NodeTypeOptimizationBarrier: true,
	graphimpl.NodeTypeSchedulingBarrier:   true,
	graphimpl.NodeTypeAllReduce:           true,
	graphimpl.NodeTypeWhile:               true,
	graphimpl.NodeTypeIf:                  true,
	graphimpl.NodeTypeSort:                true,
	graphimpl.NodeTypeCall:                true,
	graphimpl.NodeTypeDynamicUpdateSlice:  true,
	graphimpl.NodeTypeScatterSum:          true,
	graphimpl.NodeTypeScatterMin:          true,
	graphimpl.NodeTypeScatterMax:          true,
	graphimpl.NodeTypeSelectAndScatterMax: true,
	graphimpl.NodeTypeSelectAndScatterMin: true,
}

// AutoRematerialize inserts framework-chosen checkpoint boundaries until the
// static peak fits budgetBytes, then returns a fresh plan.
//
// The graph must still be building and gradients must not have been constructed
// yet: the inserted boundaries reuse the same Checkpoint/StopCheckpoint mechanism
// as manual boundaries, so gradient numerics stay on the existing, tested path.
//
// If the target cannot be reached, a RematerializationError is returned together
// with the best plan achievable; the graph is left with the boundaries inserted
// so far, which only reduce peak and never change computed values.
func AutoRematerialize(g *graphimpl.GraphImpl, budgetBytes int64, outputs []*graphimpl.NodeImpl) (*RematerializationResult, error) {
	if budgetBytes <= 0 {
		return nil, errors.WithStack(&PlanError{
			Op:      "auto-rematerialization",
			Message: "a positive device budget is required to drive automatic rematerialization",
		})
	}

	result := &RematerializationResult{}
	plan, err := Build(g, Options{BudgetBytes: 0, Outputs: outputs})
	if err != nil {
		return nil, err
	}
	if plan.PeakBytes <= budgetBytes {
		result.Plan = plan
		return result, nil
	}

	// Track an already-inserted boundary set so the search terminates.
	marked := setsMakeNodeID()
	maxIterations := len(g.Nodes()) + 1
	for iteration := 0; iteration < maxIterations; iteration++ {
		candidate := pickRematerializationCandidate(g, plan, marked)
		if candidate == nil {
			break
		}
		insertCheckpoint(g, candidate)
		marked.Insert(candidate.Id())
		result.Boundaries = append(result.Boundaries, candidate.Id())
		result.Applied = true

		plan, err = Build(g, Options{BudgetBytes: 0, Outputs: outputs})
		if err != nil {
			return result, err
		}
		if plan.PeakBytes <= budgetBytes {
			result.Plan = plan
			return result, nil
		}
	}

	// Irreducible: report the best plan with an explicit error, never a silent pass.
	rematErr := newRematError(g, plan, budgetBytes, marked)
	result.Plan = plan
	return result, rematErr
}

// pickRematerializationCandidate chooses the highest-value boundary: the largest
// static intermediate tensor whose reconstruction chain is rematerializable and
// which is not already a boundary.
func pickRematerializationCandidate(g *graphimpl.GraphImpl, plan *Plan, marked sets.Set[graphimpl.NodeId]) *graphimpl.NodeImpl {
	candidates := make([]*TensorRecord, 0)
	for _, rec := range plan.Tensors {
		if rec.IsDynamic || rec.AliasGroup >= 0 || rec.IsPersistent() {
			continue
		}
		node := g.NodeById(rec.NodeId)
		if node == nil || node.IsCheckpointed() || marked.Has(node.Id()) {
			continue
		}
		if !chainRematerializable(node) {
			continue
		}
		candidates = append(candidates, rec)
	}
	if len(candidates) == 0 {
		return nil
	}
	slices.SortFunc(candidates, func(a, b *TensorRecord) int {
		if a.Bytes != b.Bytes {
			return int(b.Bytes - a.Bytes)
		}
		return int(a.NodeId - b.NodeId)
	})
	return g.NodeById(candidates[0].NodeId)
}

// chainRematerializable walks the input chain and rejects the candidate when a
// genuinely non-reconstructible operator (stateful, ordering-sensitive or
// control-flow) sits between it and the nearest existing checkpoint boundary.
// Parameters and constants are valid reconstruction roots, not blockers: a
// rematerialized chain is allowed to terminate at them.
func chainRematerializable(start *graphimpl.NodeImpl) bool {
	visited := setsMakeNodeID()
	var walk func(n *graphimpl.NodeImpl) bool
	walk = func(n *graphimpl.NodeImpl) bool {
		if n == nil || visited.Has(n.Id()) {
			return true
		}
		visited.Insert(n.Id())
		if n.IsCheckpointed() {
			return true
		}
		// Reconstruction roots: anything below them is not part of the
		// rematerialized intermediate chain.
		if n.Type() == graphimpl.NodeTypeParameter || n.Type() == graphimpl.NodeTypeConstant {
			return true
		}
		if nonRematerializableTypes[n.Type()] {
			return false
		}
		for _, in := range n.Inputs() {
			if !walk(in) {
				return false
			}
		}
		return true
	}
	for _, in := range start.Inputs() {
		if !walk(in) {
			return false
		}
	}
	return true
}

// insertCheckpoint routes every current consumer of node through a new
// OptimizationBarrier carrying the checkpoint flags, mirroring Node.Checkpoint().
func insertCheckpoint(g *graphimpl.GraphImpl, node *graphimpl.NodeImpl) {
	barrier := graphimpl.OptimizationBarrier(node)
	barrier.SetCheckpointed(true, true)
	node.MarkNeedsRematerialization()

	// Re-point every later node that reads node directly at the barrier. The
	// barrier node itself reads node, so it is excluded by id comparison.
	for _, consumer := range g.Nodes() {
		if consumer.Id() <= barrier.Id() {
			continue
		}
		for idx, in := range consumer.Inputs() {
			if in != nil && in.Id() == node.Id() {
				consumer.RewireInput(idx, barrier)
			}
		}
	}
}

// newRematError builds the terminal error with scope and tensor diagnostics.
func newRematError(g *graphimpl.GraphImpl, plan *Plan, budget int64, marked sets.Set[graphimpl.NodeId]) error {
	reason := "no rematerializable candidate remains"
	if len(marked) == 0 {
		reason = "chain contains a non-rematerializable operator or every candidate is already a boundary"
	}
	largestScope := ""
	var largestBytes int64
	if len(plan.ScopeUsage) > 0 {
		largestScope = plan.ScopeUsage[0].ScopePath
		largestBytes = plan.ScopeUsage[0].Bytes
	}
	return errors.WithStack(&RematerializationError{
		BudgetBytes:       budget,
		Irreducible:       plan.PeakBytes,
		OverBytes:         plan.PeakBytes - budget,
		LargestScope:      largestScope,
		LargestScopeBytes: largestBytes,
		TopTensors:        plan.topTensors(5),
		Reason:            reason,
	})
}

// setsMakeNodeID is a small constructor kept local so the planner reads without
// the generic set import noise at call sites.
func setsMakeNodeID() sets.Set[graphimpl.NodeId] {
	return sets.Make[graphimpl.NodeId]()
}
