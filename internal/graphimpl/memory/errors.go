// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package memory

import (
	"fmt"

	"github.com/pkg/errors"
)

// PlanError is the base error type for build-time memory planning failures.
//
// Messages only mention shapes, scope paths and byte counts; tensor values,
// weight file paths and external addresses are never included.
type PlanError struct {
	Op      string
	Message string
}

func (e *PlanError) Error() string {
	return fmt.Sprintf("memory planning (%s): %s", e.Op, e.Message)
}

func newPlanError(format string, args ...any) error {
	return errors.WithStack(&PlanError{Op: "plan", Message: fmt.Sprintf(format, args...)})
}

// NewPlanErrorPublic builds a plan error for use by the public wrapper package.
func NewPlanErrorPublic(format string, args ...any) error {
	return newPlanError(format, args...)
}

func newPlanErrorf(format string, args ...any) error {
	return newPlanError(format, args...)
}

// AliasConflictError reports an impossible alias grouping (same underlying
// storage treated as two independent tensors, or alias members asked to share a
// slot). The planner keeps the conservative whole-tensor peak rather than
// returning a layout that looks successful.
type AliasConflictError struct {
	Reason        AliasReason
	Ref           TensorRef
	ExpectedBytes int64
	ActualBytes   int64
}

func (e *AliasConflictError) Error() string {
	return fmt.Sprintf(
		"memory planning (alias): conflict in alias group (%s) at node #%d output %d: "+
			"members disagree on storage size (%d bytes vs %d bytes); keeping conservative peak without aliasing",
		e.Reason, e.Ref.NodeId, e.Ref.OutputIndex, e.ExpectedBytes, e.ActualBytes)
}

// BudgetExceededError is returned at graph build time when the planned peak does
// not fit the declared device budget. It carries the overshoot, the largest
// contributing scope and the irreducible peak, so the problem never reaches
// execution as an out-of-memory failure.
type BudgetExceededError struct {
	// BudgetBytes is the declared device limit.
	BudgetBytes int64

	// PeakBytes is the peak that violated the budget (exact, or lower-bound
	// when dynamic tensors make the exact figure unknowable).
	PeakBytes int64

	// OverBytes is PeakBytes - BudgetBytes.
	OverBytes int64

	// LargestScope is the scope path contributing the most static bytes.
	LargestScope string

	// LargestScopeBytes is the static byte usage attributed to LargestScope.
	LargestScopeBytes int64

	// LowerBound is true when the violation is proven from the dynamic lower
	// bound rather than an exact peak.
	LowerBound bool

	// TopTensors lists the largest irreducible tensor outputs, by static bytes.
	TopTensors []TopTensor
}

// TopTensor summarizes one of the largest tensor outputs for budget diagnostics.
type TopTensor struct {
	NodeId      int64
	OutputIndex int
	ScopePath   string
	Bytes       int64
	ShapeText   string
}

func (e *BudgetExceededError) Error() string {
	basis := "exact static peak"
	if e.LowerBound {
		basis = "known lower-bound peak (runtime-dimension tensors included at their lower bound)"
	}
	return fmt.Sprintf(
		"memory planning (budget): %s %d bytes exceeds device budget %d bytes by %d bytes; "+
			"largest scope %q accounts for %d bytes; irreducible peak is %d bytes",
		basis, e.PeakBytes, e.BudgetBytes, e.OverBytes,
		e.LargestScope, e.LargestScopeBytes, e.PeakBytes)
}

func newBudgetExceededError(p *Plan, budget, peak int64, lowerBound bool) error {
	e := &BudgetExceededError{
		BudgetBytes: budget,
		PeakBytes:   peak,
		OverBytes:   peak - budget,
		LowerBound:  lowerBound,
	}
	if len(p.ScopeUsage) > 0 {
		e.LargestScope = p.ScopeUsage[0].ScopePath
		e.LargestScopeBytes = p.ScopeUsage[0].Bytes
	}
	e.TopTensors = p.topTensors(5)
	return errors.WithStack(e)
}

// RematerializationError reports that automatic rematerialization could not
// drive the graph under budget: either every candidate is already a boundary or
// the chain contains an operator that cannot be rematerialized.
type RematerializationError struct {
	BudgetBytes       int64
	Irreducible       int64
	OverBytes         int64
	LargestScope      string
	LargestScopeBytes int64
	TopTensors        []TopTensor
	Reason            string
}

func (e *RematerializationError) Error() string {
	return fmt.Sprintf(
		"memory planning (auto-rematerialization): %s; irreducible peak %d bytes still exceeds "+
			"budget %d bytes by %d bytes; largest scope %q",
		e.Reason, e.Irreducible, e.BudgetBytes, e.OverBytes, e.LargestScope)
}

// AsBudgetExceeded extracts a BudgetExceededError from err, if present.
func AsBudgetExceeded(err error) (*BudgetExceededError, bool) {
	var target *BudgetExceededError
	if errors.As(err, &target) {
		return target, true
	}
	return nil, false
}

// AsAliasConflict extracts an AliasConflictError from err, if present.
func AsAliasConflict(err error) (*AliasConflictError, bool) {
	var target *AliasConflictError
	if errors.As(err, &target) {
		return target, true
	}
	return nil, false
}

// AsRematerialization extracts a RematerializationError from err, if present.
func AsRematerialization(err error) (*RematerializationError, bool) {
	var target *RematerializationError
	if errors.As(err, &target) {
		return target, true
	}
	return nil, false
}
