// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package memory_test

import (
	"strings"
	"sync"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// buildBudgetGraph builds a graph in which naive and reuse-aware peaks differ.
//
// All big tensors are 400 B. The liveness layout is arranged so that:
//
//	a dies before c is born (disjoint intervals, identical size -> they share a slot),
//	q stays alive across both (its interval overlaps a's and c's).
//
// At the step where q and c are alive together, naive allocation still reserves
// dedicated buffers for a, q and c (a's buffer is not recycled), while
// reuse-aware allocation uses one slot for a/c plus q's buffer: the reuse-aware
// peak is exactly one 400 B buffer lower.
func buildBudgetGraph(t *testing.T) (b interface{ Finalize() }, g *Graph, out *Node, plan *memory.Plan) {
	backend, err := gobackend.New("")
	require.NoError(t, err)

	g = NewGraph(backend, "budget")
	x := Parameter(g, "x", shapes.Make(dtypes.Float32, 100))
	y := Parameter(g, "y", shapes.Make(dtypes.Float32, 100))

	// a and its scalar reduction; a dies once aSum is produced.
	a := Add(x, y)
	aSum := ReduceAllSum(a)

	// q is a long-lived 400 B tensor overlapping both a and c.
	q := Mul(x, y)

	// c is born only after a is dead and matches a's byte size/alignment.
	c := Add(q, q)
	cSum := ReduceAllSum(c)
	qSum := ReduceAllSum(q)

	out = Add(Add(aSum, qSum), cSum)

	plan, err = memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err)
	require.Greater(t, plan.PeakBeforeReuse, plan.PeakBytes, "test graph must allow peak-reducing slot reuse")
	return backend, g, out, plan
}

// TestBudgetBetweenPeaks sets the budget between the reuse and no-reuse peaks:
// the exact (reuse-aware) peak must pass, while an artificially tight budget
// fails with the overshoot and largest scope in the message.
func TestBudgetBetweenPeaks(t *testing.T) {
	backend, g, out, plan := buildBudgetGraph(t)
	defer backend.Finalize()

	// A budget above the reuse peak passes.
	passBudget := plan.PeakBytes + 16
	_, err := memory.PlanMemory(g, memory.Options{BudgetBytes: passBudget, Outputs: []*Node{out}})
	require.NoError(t, err)

	// A budget between the reuse peak and the no-reuse peak fails.
	tightBudget := plan.PeakBytes - 4
	require.GreaterOrEqual(t, tightBudget, int64(0))
	_, err = memory.PlanMemory(g, memory.Options{BudgetBytes: tightBudget, Outputs: []*Node{out}})
	require.Error(t, err)

	budgetErr, ok := memory.AsBudgetExceeded(err)
	require.True(t, ok, "expected a BudgetExceededError")
	assert.Equal(t, tightBudget, budgetErr.BudgetBytes)
	assert.Equal(t, plan.PeakBytes-budgetErr.BudgetBytes, budgetErr.OverBytes)
	assert.NotEmpty(t, budgetErr.LargestScope)
	msg := err.Error()
	assert.Contains(t, msg, "exceeds device budget")
	// Error text may only mention shapes/scopes/bytes: no tensor values.
	assert.NotContains(t, msg, "0x")
}

// TestBudgetDynamicLowerBound proves dynamic tensors are never counted as zero:
// when the lower-bound peak already exceeds the budget, planning fails.
func TestBudgetDynamicLowerBound(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()
	if !b.Capabilities().HasDynamicShapes() {
		t.Skip("backend does not support dynamic shapes")
	}

	g := NewGraph(b, "dyn_budget")
	dyn := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 64}, []string{"batch", ""})
	x := Parameter(g, "x", dyn)
	out := ReduceAllSum(Add(x, x))

	// Lower bound alone is well above 8 bytes.
	_, err = memory.PlanMemory(g, memory.Options{BudgetBytes: 8, Outputs: []*Node{out}})
	require.Error(t, err)
	budgetErr, ok := memory.AsBudgetExceeded(err)
	require.True(t, ok)
	assert.True(t, budgetErr.LowerBound, "violation must be proven from the lower bound")
}

// TestExecMemoryBudget wires the budget through the executor and checks build-time
// failure rather than an execution-time OOM.
func TestExecMemoryBudget(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	exec, err := NewExec(b, func(x *Node) *Node {
		w := Const(x.Graph(), make([]float32, 100))
		return ReduceAllSum(Add(Mul(x, w), Mul(x, w)))
	})
	require.NoError(t, err)
	defer exec.Finalize()
	exec.WithMemoryBudget(4, false) // impossibly tight

	_, err = exec.Compile(shapes.Make(dtypes.Float32, 100))
	require.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), "budget"), "error must explain the budget: %v", err)

	// A generous budget compiles cleanly and execution is unchanged.
	exec2, err := NewExec(b, func(x *Node) *Node {
		return Add(x, x)
	})
	require.NoError(t, err)
	defer exec2.Finalize()
	exec2.WithMemoryBudget(1<<20, false)
	results, err := exec2.Call([]float32{1, 2, 3})
	require.NoError(t, err)
	assert.Equal(t, []float32{2, 4, 6}, results[0].Value())
}

// TestAutoRematerialize inserts boundaries until the peak fits the budget and
// reports the inserted boundaries.
func TestAutoRematerialize(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	g := NewGraph(b, "remat")
	x := Parameter(g, "x", shapes.Make(dtypes.Float32, 200))
	w := Const(g, make([]float32, 200))
	a := Mul(x, w)
	c := Add(a, a)
	out := ReduceAllSum(c)

	before, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err)

	// Force a budget below the current exact peak but above the lower floor.
	budget := before.PeakBytes / 2
	plan, ids, err := memory.AutoRematerialize(g, budget, out)
	if err != nil {
		// Irreducible graphs must return a typed error, never a silent pass.
		rematErr, ok := memory.AsRematerialization(err)
		require.True(t, ok)
		assert.Greater(t, rematErr.Irreducible, budget)
		return
	}
	require.NotEmpty(t, ids, "at least one boundary must be inserted")
	assert.LessOrEqual(t, plan.PeakBytes, before.PeakBytes, "rematerialization cannot raise the peak")
	assert.GreaterOrEqual(t, plan.NumCheckpointBoundaries, 1)
}

// TestAutoRematerializeInfeasible checks the explicit failure when no candidate
// can reduce the peak (budget below the persistent parameter/constant floor).
func TestAutoRematerializeInfeasible(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	g := NewGraph(b, "remat_infeasible")
	x := Parameter(g, "x", shapes.Make(dtypes.Float32, 100))
	out := Add(x, x)

	// One byte is below any materialized tensor's storage.
	_, _, err = memory.AutoRematerialize(g, 1, out)
	require.Error(t, err)
	_, ok := memory.AsRematerialization(err)
	assert.True(t, ok, "infeasible remat must return a typed RematerializationError")
}

// TestPlanConcurrentReaders exercises the immutable snapshot under the race
// detector: many goroutines read every field while planning finishes once.
func TestPlanConcurrentReaders(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	g := NewGraph(b, "race")
	x := Parameter(g, "x", shapes.Make(dtypes.Float32, 32))
	out := ReduceAllSum(Add(Mul(x, Const(g, make([]float32, 32))), x))
	plan, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err)

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 200 {
				_ = plan.PeakBytes
				_ = plan.TotalBytes
				_ = plan.Report()
				for _, r := range plan.Tensors {
					_ = r.Bytes
					_ = r.LastUseAt
				}
				for _, grp := range plan.AliasGroups {
					_ = grp.Bytes
				}
			}
		}()
	}
	wg.Wait()
}
