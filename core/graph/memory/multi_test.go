// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package memory_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/memory"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/internal/graphbridge"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMultiGraphSharedConstantDedup plans two graphs sharing one external buffer
// and asserts the aggregate counts that constant's storage only once.
func TestMultiGraphSharedConstantDedup(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	shared := tensors.MustFromAnyValue(make([]float32, 16))
	buf, err := shared.Buffer(backend, 0)
	require.NoError(t, err)

	buildOne := func(name string) (*Graph, *Node) {
		g := NewGraph(backend, name)
		c := graphbridge.ConstBuffer(g, buf, false)
		x := Parameter(g, "x", shapes.Make(dtypes.Float32, 16))
		out := ReduceAllSum(Add(c, x))
		return g, out
	}

	g1, out1 := buildOne("g1")
	g2, out2 := buildOne("g2")
	p1, err := memory.PlanMemory(g1, memory.Options{Outputs: []*Node{out1}})
	require.NoError(t, err)
	p2, err := memory.PlanMemory(g2, memory.Options{Outputs: []*Node{out2}})
	require.NoError(t, err)

	mp := memory.AggregatePlans(p1, p2)
	require.Len(t, mp.Plans, 2)

	// The shared constant contributes 64 bytes per graph naively, but the
	// aggregate must de-duplicate one copy.
	const sharedBytes = int64(16 * 4)
	assert.Equal(t, sharedBytes, mp.SharedConstantDedupBytes)
	assert.Equal(t, p1.TotalBytes+p2.TotalBytes-sharedBytes, mp.CombinedTotalBytes)
	assert.Equal(t, p1.PeakBytes+p2.PeakBytes-sharedBytes, mp.CombinedPeakBytes)
	assert.Contains(t, mp.Report(), "Multi-graph memory plan")
}

// TestPlanExecGraphs drives two input shapes through one executor and checks the
// aggregate view covers both cached graphs.
func TestPlanExecGraphs(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	exec, err := NewExec(backend, func(x *Node) *Node {
		return Add(x, x)
	})
	require.NoError(t, err)
	defer exec.Finalize()

	_, err = exec.Call([]float32{1, 2, 3})
	require.NoError(t, err)
	_, err = exec.Call([]float32{1, 2, 3, 4})
	require.NoError(t, err)
	require.Equal(t, 2, len(exec.CachedGraphs()))

	mp, err := memory.PlanExecGraphs(exec)
	require.NoError(t, err)
	require.Len(t, mp.Plans, 2)
	assert.Greater(t, mp.CombinedTotalBytes, int64(0))
}

// TestErrorContentLeakSafety verifies planning errors never embed tensor values,
// weight paths or external addresses: only shapes, scopes and byte counts.
func TestErrorContentLeakSafety(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	g := NewGraph(backend, "leak_safety")
	x := Parameter(g, "x", shapes.Make(dtypes.Float32, 8))
	// A distinctive tensor value that must never appear in an error message.
	secret := Const(g, []float32{999.25, 42.5, 7.75, 1.125, 3.5, 6.0, 8.0, 2.0})
	out := ReduceAllSum(Add(Mul(x, secret), secret))

	_, err = memory.PlanMemory(g, memory.Options{BudgetBytes: 1, Outputs: []*Node{out}})
	require.Error(t, err)
	msg := err.Error()
	assert.NotContains(t, msg, "999.25")
	assert.NotContains(t, msg, "42.5")
	assert.Contains(t, msg, "budget")
}
