// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package memory provides build-time tensor memory planning for GoMLX graphs.
//
// A [Plan] answers, before any execution starts, how many tensor outputs are
// simultaneously alive in a graph, what the peak storage is, which tensors can
// share one storage slot (and the peak before vs. after reuse), which tensors
// alias the same underlying storage (counted once and never co-slotted), and
// whether a declared device budget is exceeded.
//
// Planning is a read-only operation over a single immutable snapshot: it does not
// start background tasks, adds no dependencies, and concurrent reads of a Plan
// are data-race free. Without an explicit budget the compile/execute/cache path
// is byte-for-byte unchanged; planning reports only add information.
package memory

import (
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/internal/graphbridge"
	"github.com/gomlx/gomlx/internal/graphimpl"
	impl "github.com/gomlx/gomlx/internal/graphimpl/memory"
)

func init() {
	graphimpl.RegisterMemoryPlanner(
		func(g *graphimpl.GraphImpl, budgetBytes int64, outputs []*graphimpl.NodeImpl) (graphimpl.MemoryPlanResult, error) {
			return impl.Build(g, impl.Options{BudgetBytes: budgetBytes, Outputs: outputs})
		},
		func(g *graphimpl.GraphImpl, budgetBytes int64, outputs []*graphimpl.NodeImpl) (graphimpl.MemoryPlanResult, []graphimpl.NodeId, error) {
			result, err := impl.AutoRematerialize(g, budgetBytes, outputs)
			if result == nil {
				return nil, nil, err
			}
			return result.Plan, result.Boundaries, err
		},
	)
}

// TensorRef identifies one tensor output within a graph.
type TensorRef = impl.TensorRef

// AliasReason explains why tensors were placed in the same alias group.
type AliasReason = impl.AliasReason

// Alias group reasons.
const (
	AliasBitcast          = impl.AliasBitcast
	AliasExternalBuffer   = impl.AliasExternalBuffer
	AliasSharedConstant   = impl.AliasSharedConstant
	AliasRepeatedArgument = impl.AliasRepeatedArgument
)

// TensorRecord is one materialized tensor output observed during planning.
type TensorRecord = impl.TensorRecord

// AliasGroupRecord describes a set of tensors proven to alias the same storage.
type AliasGroupRecord = impl.AliasGroupRecord

// AliasConflict describes an alias group rejected because its members could not
// provably share one storage allocation; its members are conservatively counted
// independently instead.
type AliasConflict = impl.AliasConflict

// SlotReuse records one accepted reuse-slot placement.
type SlotReuse = impl.SlotReuse

// ScopeUsage aggregates byte usage attributed to one function scope.
type ScopeUsage = impl.ScopeUsage

// TopTensor summarizes one of the largest tensor outputs.
type TopTensor = impl.TopTensor

// BudgetExceededError is returned at graph build time when the planned peak
// exceeds the declared device budget.
type BudgetExceededError = impl.BudgetExceededError

// AliasConflictError reports an impossible alias grouping.
type AliasConflictError = impl.AliasConflictError

// RematerializationError reports that automatic rematerialization could not
// drive the graph under budget.
type RematerializationError = impl.RematerializationError

// Options controls a planning pass.
type Options struct {
	// BudgetBytes, when > 0, enables the hard device budget.
	BudgetBytes int64

	// Outputs are the graph outputs; output tensors are live for the whole graph.
	// When nil, outputs recorded during Graph.Compile are used.
	Outputs []*graph.Node
}

// Plan is an immutable build-time memory planning snapshot.
type Plan = impl.Plan

// MultiPlan is the aggregate memory view of several per-shape graphs.
type MultiPlan = impl.MultiPlan

// AggregatePlans combines independently computed per-graph plans into one
// executor-level view, de-duplicating constants shared across graphs.
func AggregatePlans(plans ...*Plan) *MultiPlan {
	return impl.AggregatePlans(plans...)
}

// PlanMemory runs a full planning pass over g without mutating it.
//
// The graph must still be building (not compiled). When opts.BudgetBytes > 0 and
// the exact peak (or the known dynamic lower bound) exceeds the budget, a
// [BudgetExceededError] is returned alongside the plan so callers can inspect
// both the failure and the conservative figures.
func PlanMemory(g *graph.Graph, opts Options) (*Plan, error) {
	if g == nil {
		return nil, impl.NewPlanErrorPublic("cannot plan a nil graph")
	}
	internalOpts := impl.Options{
		BudgetBytes: opts.BudgetBytes,
		Outputs:     graphbridge.NodeImpls(opts.Outputs),
	}
	return impl.Build(graphbridge.GraphImpl(g), internalOpts)
}

// AutoRematerialize inserts framework-chosen checkpoint boundaries until the
// static peak fits budgetBytes, then returns the resulting plan and the inserted
// boundary ids.
//
// It must be called before gradients are constructed: the inserted boundaries
// reuse the same mechanism as [graph.Node.Checkpoint], keeping gradient numerics
// on the existing tested path. When the budget cannot be reached, a
// [RematerializationError] is returned with the irreducible peak and the largest
// tensor outputs.
func AutoRematerialize(g *graph.Graph, budgetBytes int64, outputs ...*graph.Node) (*Plan, []graph.NodeId, error) {
	if g == nil {
		return nil, nil, impl.NewPlanErrorPublic("cannot rematerialize a nil graph")
	}
	result, err := impl.AutoRematerialize(graphbridge.GraphImpl(g), budgetBytes, graphbridge.NodeImpls(outputs))
	if result == nil {
		return nil, nil, err
	}
	ids := make([]graph.NodeId, 0, len(result.Boundaries))
	for _, id := range result.Boundaries {
		ids = append(ids, graph.NodeId(id))
	}
	return result.Plan, ids, err
}

// AsBudgetExceeded extracts a BudgetExceededError from err.
func AsBudgetExceeded(err error) (*BudgetExceededError, bool) {
	return impl.AsBudgetExceeded(err)
}

// AsAliasConflict extracts an AliasConflictError from err.
func AsAliasConflict(err error) (*AliasConflictError, bool) {
	return impl.AsAliasConflict(err)
}

// AsRematerialization extracts a RematerializationError from err.
func AsRematerialization(err error) (*RematerializationError, bool) {
	return impl.AsRematerialization(err)
}

// ShapeOf is a convenience accessor for tests and reports.
func ShapeOf(rec *TensorRecord) shapes.Shape { return rec.Shape }

// PlanExecGraphs plans every graph currently cached in e and returns the
// executor-level aggregate view. Each distinct input shape is planned on its own
// snapshot; constants sharing storage across the cached graphs are counted once.
func PlanExecGraphs(e *graph.Exec) (*MultiPlan, error) {
	cached := e.CachedGraphs()
	plans := make([]*Plan, 0, len(cached))
	for _, g := range cached {
		plan, err := PlanMemory(g, Options{})
		if err != nil {
			return nil, err
		}
		plans = append(plans, plan)
	}
	return AggregatePlans(plans...), nil
}
