// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package memory_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/autodiff"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/memory"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// buildGradientChain builds a forward+backward graph on the shared backend. When
// checkpoint is true, the long-lived intermediate h2 is wrapped in a manual
// checkpoint boundary, exercising the existing Checkpoint/StopCheckpoint path.
func buildGradientChain(t *testing.T, b compute.Backend, checkpoint bool) (*Graph, *Node, *Node) {
	g := NewGraph(b, "grad_chain")
	x := Const(g, []float32{1, 2, 3, 4})
	w := Parameter(g, "w", shapes.Make(dtypes.Float32, 4))

	h1 := Mul(w, x)
	h2 := Add(h1, x)
	if checkpoint {
		// Boundary: h2's forward value is not retained for the backward pass;
		// it is reconstructed lazily during reverse-mode autodiff.
		h2 = h2.Checkpoint()
	}
	h3 := Mul(h2, h2)
	loss := ReduceAllSum(h3)
	grads := autodiff.Gradient(loss, w)
	require.Len(t, grads, 1)
	return g, loss, grads[0]
}

// TestCheckpointGradientConsistencyAndPeak builds the same model with and without
// a manual checkpoint boundary on a shared backend, then asserts:
//  1. both plans are produced from the same interval/byte/alias basis, and
//  2. the computed loss and gradients match within tolerance (the rematerialized
//     reconstruction must be numerically identical to the retained forward value).
func TestCheckpointGradientConsistencyAndPeak(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	gPlain, lossPlain, gradPlain := buildGradientChain(t, backend, false)
	gCheck, lossCheck, gradCheck := buildGradientChain(t, backend, true)

	planPlain, err := memory.PlanMemory(gPlain, memory.Options{Outputs: []*Node{lossPlain, gradPlain}})
	require.NoError(t, err)
	planCheck, err := memory.PlanMemory(gCheck, memory.Options{Outputs: []*Node{lossCheck, gradCheck}})
	require.NoError(t, err)

	// Both plans share the same accounting basis and report a valid peak.
	assert.Greater(t, planPlain.PeakBytes, int64(0))
	assert.Greater(t, planCheck.PeakBytes, int64(0))
	assert.GreaterOrEqual(t, planCheck.NumCheckpointBoundaries, 1, "checkpoint graph must count its boundary")
	assert.Equal(t, planPlain.NumDynamic, planCheck.NumDynamic)

	// Numerics stay identical along the existing gradient path.
	require.NoError(t, gPlain.Compile(lossPlain, gradPlain))
	require.NoError(t, gCheck.Compile(lossCheck, gradCheck))
	wInit := tensors.MustFromAnyValue([]float32{0.5, 1.5, 2.5, 3.5})
	outPlain := gPlain.Run(wInit)
	outCheck := gCheck.Run(wInit)
	require.Len(t, outPlain, 2)
	require.Len(t, outCheck, 2)
	assert.True(t, outPlain[0].InDelta(outCheck[0], 1e-5), "loss values must match")
	assert.True(t, outPlain[1].InDelta(outCheck[1], 1e-5), "gradients must match within tolerance")
}

// TestAutoRematerializationInsertsBoundaries verifies the framework-driven
// path inserts checkpoint boundaries when the budget is tightened, and either
// converges or returns a typed RematerializationError naming the irreducible
// peak — never a silent success.
func TestAutoRematerializationInsertsBoundaries(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	g := NewGraph(backend, "auto_remat")
	x := Parameter(g, "x", shapes.Make(dtypes.Float32, 64))
	w := Const(g, make([]float32, 64))
	h1 := Mul(x, w)
	h2 := Add(h1, h1)
	h3 := Mul(h2, h2)
	loss := ReduceAllSum(h3)

	// A second independent branch whose big intermediate is disjoint in
	// liveness from h3 and identical in size, so slot reuse is available.
	side := Add(x, w)
	sideSum := ReduceAllSum(side)
	loss = Add(loss, sideSum)

	plan0, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{loss}})
	require.NoError(t, err)
	// There must be a rematerializable intermediate before we tighten the budget.
	require.Greater(t, plan0.PeakBeforeReuse, plan0.PeakBytes,
		"the model must admit reuse so a tighter budget has somewhere to act")

	// Tighten below the reuse-aware peak to force boundary insertion.
	budget := plan0.PeakBytes - 1
	require.Greater(t, budget, int64(0))
	plan1, ids, err := memory.AutoRematerialize(g, budget, loss)

	if err == nil {
		// Converged: boundaries were inserted and the reuse peak fits.
		require.NotEmpty(t, ids, "convergence below the original peak requires a boundary")
		assert.LessOrEqual(t, plan1.PeakBytes, budget)
		assert.GreaterOrEqual(t, plan1.NumCheckpointBoundaries, 1)
		return
	}

	// Otherwise the failure must be explicit and carry the irreducible figures.
	rematErr, ok := memory.AsRematerialization(err)
	require.True(t, ok, "infeasible auto-remat must return a typed error")
	assert.Greater(t, rematErr.Irreducible, budget)
	assert.NotEmpty(t, rematErr.LargestScope)
}

// TestManualCheckpointStillWorks guards the existing manual Checkpoint path and
// confirms planning recognizes its boundary without breaking execution.
func TestManualCheckpointStillWorks(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	g := NewGraph(backend, "manual_checkpoint")
	xVal := Const(g, []float32{1, 2, 3})
	w := Parameter(g, "w", shapes.Make(dtypes.Float32, 3))

	checkpointX := xVal.Checkpoint()
	x := Mul(checkpointX, w)
	y := Add(x, checkpointX)
	checkpointY := y.StopCheckpoint()
	loss := ReduceAllSum(checkpointY)
	grads := autodiff.Gradient(loss, w)

	plan, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{loss, grads[0]}})
	require.NoError(t, err)
	assert.GreaterOrEqual(t, plan.NumCheckpointBoundaries, 1, "manual boundaries must be counted")

	require.NoError(t, g.Compile(loss, grads[0]))
	wInit := tensors.MustFromAnyValue([]float32{0.5, 1.5, 2.5})
	results := g.Run(wInit)
	require.Len(t, results, 2)
	assert.NotNil(t, results[0].Value())
	assert.NotNil(t, results[1].Value())
}
