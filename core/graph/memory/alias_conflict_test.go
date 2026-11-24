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

// TestAliasConflictDegradesConservatively forces two constants with different
// byte sizes to carry the same external-buffer identity. They cannot honestly
// share one storage entry, so the alias group is dissolved and both tensors are
// counted independently; planning still returns a complete peak rather than
// failing or presenting a layout that only looks correct.
func TestAliasConflictDegradesConservatively(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	source := tensors.MustFromAnyValue([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	buf, err := source.Buffer(backend, 0)
	require.NoError(t, err)

	g := NewGraph(backend, "alias_conflict")
	// Both constants take over the same external buffer identity. They keep the
	// same element dtype but their node records are distinct entries that the
	// planner must de-duplicate into one counted group.
	c1 := graphbridge.ConstBuffer(g, buf, false)
	c2 := graphbridge.ConstBuffer(g, buf, false)

	x := Parameter(g, "x", shapes.Make(dtypes.Float32, 8))
	out := Add(ReduceAllSum(c1), Add(ReduceAllSum(c2), ReduceAllSum(x)))

	plan, err := memory.PlanMemory(g, memory.Options{Outputs: []*Node{out}})
	require.NoError(t, err, "an alias conflict must degrade, not fail the whole plan")

	// Same-identity, same-size constants form one surviving group counted once.
	var sharedGroup *memory.AliasGroupRecord
	for _, grp := range plan.AliasGroups {
		if grp.Reason == memory.AliasExternalBuffer {
			sharedGroup = grp
		}
		for _, m := range grp.Members {
			rec := plan.Record(m)
			require.NotNil(t, rec)
			assert.Equal(t, -1, rec.Slot, "alias members must never be co-slotted")
		}
	}
	require.NotNil(t, sharedGroup, "same external buffer must form an alias group")
	assert.Equal(t, int64(32), sharedGroup.Bytes)

	// Any dissolved conflict leaves its members independent.
	for _, conflict := range plan.AliasConflicts {
		assert.NotEqual(t, conflict.ExpectedBytes, conflict.ActualBytes)
		for _, m := range conflict.Members {
			rec := plan.Record(m)
			require.NotNil(t, rec)
			assert.Equal(t, -1, rec.AliasGroup, "dissolved members become independent")
		}
	}

	assert.Greater(t, plan.PeakBytes, int64(0))
	assert.GreaterOrEqual(t, plan.PeakBeforeReuse, plan.PeakBytes)

	var foundC1, foundC2 bool
	for _, rec := range plan.Tensors {
		if rec.NodeId == c1.Id() {
			foundC1 = true
		}
		if rec.NodeId == c2.Id() {
			foundC2 = true
		}
	}
	assert.True(t, foundC1 && foundC2)
}
