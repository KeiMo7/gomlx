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

// TestPlanIntervalsAndPeak builds a small DAG and checks liveness intervals and
// the before/after reuse peak figures.
func TestPlanIntervalsAndPeak(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	g := NewGraph(b, "intervals")
	x := Parameter(g, "x", shapes.Make(dtypes.Float32, 4)) // 16 bytes
	w := Const(g, []float32{1, 2, 3, 4})                   // 16 bytes, persistent
	a := Mul(x, w)                                         // 16 bytes
	c := Add(a, a)                                         // 16 bytes, last consumer of a
	out := ReduceAllSum(c)                                 // 4 bytes, output

	plan, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err)
	require.NotNil(t, plan)

	// Every materialized output appears in the plan.
	assert.GreaterOrEqual(t, len(plan.Tensors), 5)

	// Locate records by node id and assert interval ordering.
	rec := func(n *Node) *memory.TensorRecord {
		for _, r := range plan.Tensors {
			if r.NodeId == n.Id() {
				return r
			}
		}
		return nil
	}
	require.NotNil(t, rec(x))
	require.NotNil(t, rec(a))
	require.NotNil(t, rec(c))

	assert.Equal(t, rec(a).BornAt, int(a.Id()))
	assert.Equal(t, rec(a).LastUseAt, int(c.Id()))
	assert.True(t, rec(x).IsPersistent())
	assert.True(t, rec(w).IsPersistent())
	assert.True(t, rec(out).IsPersistent())

	// Exact peak must be positive and never exceed the total static bytes.
	assert.Greater(t, plan.PeakBytes, int64(0))
	assert.LessOrEqual(t, plan.PeakBytes, plan.TotalBytes)
	assert.GreaterOrEqual(t, plan.PeakBeforeReuse, plan.PeakBytes)
}

// TestPlanMultiOutput checks that every output of a multi-output op is counted.
func TestPlanMultiOutput(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	g := NewGraph(b, "multi_output")
	x := Parameter(g, "x", shapes.Make(dtypes.Float32, 6))
	values, indices := TopK(x, 3, 0)
	out := Add(values, ConvertDType(indices, dtypes.Float32))

	plan, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err)

	// The multi-output node contributes two distinct records.
	count := 0
	for _, r := range plan.Tensors {
		if r.NodeId == values.Id() || r.NodeId == indices.Id() {
			count++
		}
	}
	assert.Equal(t, 2, count, "each output of a multi-output node is accounted separately")
}

// TestPlanRepeatedArgument checks that one node feeding two inputs is grouped.
func TestPlanRepeatedArgument(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	g := NewGraph(b, "repeated_arg")
	x := Parameter(g, "x", shapes.Make(dtypes.Float32, 4))
	// Add consumes x twice through distinct edges.
	y := Add(x, x)
	out := ReduceAllSum(y)

	plan, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err)

	var repeated *memory.AliasGroupRecord
	for _, grp := range plan.AliasGroups {
		if grp.Reason == memory.AliasRepeatedArgument {
			repeated = grp
			break
		}
	}
	require.NotNil(t, repeated, "repeated argument must form an alias group")
	assert.GreaterOrEqual(t, len(repeated.Members), 1)

	// Alias-group tensors never receive a reuse slot.
	for _, m := range repeated.Members {
		r := plan.Record(m)
		require.NotNil(t, r)
		assert.Equal(t, repeated.Id, r.AliasGroup)
		assert.Equal(t, -1, r.Slot, "alias members must never be co-slotted")
	}
}

// TestPlanBitcastAlias checks same-width reinterpretation is grouped once.
func TestPlanBitcastAlias(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	g := NewGraph(b, "bitcast")
	x := Parameter(g, "x", shapes.Make(dtypes.Uint32, 4))
	// Equal-width reinterpretation: uint32 -> int32 keeps the same byte width.
	y := Bitcast(x, dtypes.Int32)
	out := ReduceAllSum(ConvertDType(y, dtypes.Float32))

	plan, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err)

	var bitcastGroup *memory.AliasGroupRecord
	for _, grp := range plan.AliasGroups {
		if grp.Reason == memory.AliasBitcast {
			bitcastGroup = grp
			break
		}
	}
	require.NotNil(t, bitcastGroup, "same-width bitcast must form an alias group")
	assert.Equal(t, 2, len(bitcastGroup.Members))
	assert.Equal(t, int64(16), bitcastGroup.Bytes)
}

// TestPlanSharedConstant groups constants sharing one external buffer.
func TestPlanSharedConstant(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	g := NewGraph(b, "shared_const")
	t1 := tensors.MustFromAnyValue([]float32{1, 2, 3, 4})
	buf, err := t1.Buffer(b, 0)
	require.NoError(t, err)

	// Two constant nodes taking over the same external buffer.
	c1 := graphbridge.ConstBuffer(g, buf, false)
	c2 := graphbridge.ConstBuffer(g, buf, false)
	out := Add(ReduceAllSum(c1), ReduceAllSum(c2))

	plan, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err)

	found := false
	for _, grp := range plan.AliasGroups {
		if grp.Reason == memory.AliasExternalBuffer {
			found = true
			assert.Equal(t, int64(16), grp.Bytes)
		}
	}
	assert.True(t, found, "external-buffer constants must be de-duplicated into one group")
}

// TestPlanDynamicShape keeps runtime-dimension tensors separate and reports a
// lower bound instead of treating them as zero bytes.
func TestPlanDynamicShape(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()
	if !b.Capabilities().HasDynamicShapes() {
		t.Skip("backend does not support dynamic shapes")
	}

	g := NewGraph(b, "dynamic")
	dynShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 4}, []string{"batch", ""})
	x := Parameter(g, "x", dynShape)
	y := Add(x, x)
	out := ReduceAllSum(y)

	plan, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err)
	assert.GreaterOrEqual(t, plan.NumDynamic, 2, "parameter and its dynamic consumers are runtime-sized")

	// Dynamic tensors have no exact bytes but carry a positive lower bound.
	for _, r := range plan.Tensors {
		if r.IsDynamic {
			assert.Equal(t, int64(0), r.Bytes)
			assert.Greater(t, r.KnownLowerBoundBytes, int64(0))
			assert.Equal(t, -1, r.Slot, "dynamic tensors never reuse slots")
		}
	}
	// The lower-bound peak stays separate from the exact peak.
	assert.Greater(t, plan.LowerBoundPeakBytes, int64(0))
}

// TestPlanPackedDType verifies sub-byte types round up to whole bytes.
func TestPlanPackedDType(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	g := NewGraph(b, "packed")
	// 5 packed Int4 elements -> ceil(5*4/8) = 3 bytes.
	x := Parameter(g, "x", shapes.Make(dtypes.Int4, 5))
	out := ReduceAllSum(ConvertDType(x, dtypes.Int32))

	plan, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err)

	for _, r := range plan.Tensors {
		if r.NodeId == x.Id() {
			assert.Equal(t, int64(3), r.Bytes, "packed Int4 must round up to whole bytes")
		}
	}
}

// TestPlanDeterministic runs the same planning twice and compares every field.
func TestPlanDeterministic(t *testing.T) {
	b, err := gobackend.New("")
	require.NoError(t, err)
	defer b.Finalize()

	build := func() *Graph {
		g := NewGraph(b, "det")
		x := Parameter(g, "x", shapes.Make(dtypes.Float32, 8))
		w := Const(g, make([]float32, 8))
		a := Mul(x, w)
		c := Add(a, a)
		_ = ReduceAllSum(c)
		return g
	}

	g1 := build()
	p1, err := memory.PlanMemory(g1, memory.Options{})
	require.NoError(t, err)
	g2 := build()
	p2, err := memory.PlanMemory(g2, memory.Options{})
	require.NoError(t, err)

	assert.Equal(t, p1.TotalBytes, p2.TotalBytes)
	assert.Equal(t, p1.PeakBytes, p2.PeakBytes)
	assert.Equal(t, p1.PeakBeforeReuse, p2.PeakBeforeReuse)
	assert.Equal(t, p1.ReusedBytes, p2.ReusedBytes)
	assert.Equal(t, len(p1.Reuses), len(p2.Reuses))
	assert.Equal(t, p1.NumSlots(), p2.NumSlots())
	assert.Equal(t, p1.Report(), p2.Report())
}
