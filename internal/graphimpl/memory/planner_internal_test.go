// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package memory

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/internal/graphimpl"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDetectAliasConflictsDissolvesGroup verifies that an alias group whose
// members disagree on static byte size is dissolved into independent tensors and
// recorded as a conflict, rather than producing a misleading layout.
func TestDetectAliasConflictsDissolvesGroup(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	g := graphimpl.NewGraphImpl(backend, "conflict_internal")
	// Two constants of different sizes sharing one external identity.
	big := shapes.Make(dtypes.Float32, 8)  // 32 bytes
	small := shapes.Make(dtypes.Float32, 2) // 8 bytes
	n1 := graphimpl.Const(g, make([]float32, 8))
	n2 := graphimpl.Const(g, make([]float32, 2))
	_ = big
	_ = small

	p := &planner{
		g:            g,
		groupByKey:   make(map[aliasKey]int),
		outputNodeIDs: makeSet(n1.Id(), n2.Id()),
	}
	// Fabricate a conflicting group directly.
	group := &AliasGroupRecord{
		Id:     0,
		Reason: AliasExternalBuffer,
		Members: []TensorRef{
			{NodeId: n1.Id()},
			{NodeId: n2.Id()},
		},
	}
	p.groups = []*AliasGroupRecord{group}
	p.records = []*TensorRecord{
		{NodeId: n1.Id(), Shape: big, Bytes: 32, KnownLowerBoundBytes: 32, Alignment: 16, AliasGroup: 0, Slot: -1},
		{NodeId: n2.Id(), Shape: small, Bytes: 8, KnownLowerBoundBytes: 8, Alignment: 16, AliasGroup: 0, Slot: -1},
	}

	p.detectAliasConflicts()

	require.Len(t, p.aliasConflicts, 1, "a size mismatch must be recorded")
	assert.Equal(t, int64(32), p.aliasConflicts[0].ExpectedBytes)
	assert.Equal(t, int64(8), p.aliasConflicts[0].ActualBytes)
	assert.Empty(t, p.groups, "the conflicting group must be dissolved")
	for _, rec := range p.records {
		assert.Equal(t, -1, rec.AliasGroup, "dissolved members become independent tensors")
	}
}

func makeSet(ids ...graphimpl.NodeId) map[graphimpl.NodeId]struct{} {
	s := make(map[graphimpl.NodeId]struct{}, len(ids))
	for _, id := range ids {
		s[id] = struct{}{}
	}
	return s
}
