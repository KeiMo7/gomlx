// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package memory

import "slices"

// MultiPlan is the aggregate memory view of several graphs compiled by one
// executor for different input shapes. Each graph is planned independently;
// constants sharing one external/shared storage identity across graphs are
// counted once in the combined figures.
type MultiPlan struct {
	// Plans are the per-graph snapshots, in the order given to AggregatePlans.
	Plans []*Plan

	// CombinedPeakBytes sums each graph's reuse-aware peak after cross-graph
	// shared constants have been de-duplicated.
	CombinedPeakBytes int64

	// CombinedPeakBeforeReuse sums the conservative no-reuse peaks after
	// cross-graph de-duplication.
	CombinedPeakBeforeReuse int64

	// CombinedTotalBytes sums per-graph totals, de-duplicating shared constants.
	CombinedTotalBytes int64

	// SharedConstantDedupBytes is the storage counted once instead of once per
	// graph because it is backed by the same external/shared constant.
	SharedConstantDedupBytes int64

	// NumDynamic totals runtime-dimension tensor outputs across all graphs.
	NumDynamic int

	// NumCheckpointBoundaries totals checkpoint boundaries across all graphs.
	NumCheckpointBoundaries int
}

// AggregatePlans combines independently computed per-graph plans into one
// executor-level view, de-duplicating constants that share storage across graphs.
// The input plans are never mutated.
func AggregatePlans(plans ...*Plan) *MultiPlan {
	mp := &MultiPlan{Plans: slices.Clone(plans)}

	// Track shared constant identities already counted, so each underlying
	// storage contributes its bytes only once to the combined totals.
	seenShared := make(map[any]bool)

	for _, plan := range plans {
		if plan == nil {
			continue
		}
		mp.NumDynamic += plan.NumDynamic
		mp.NumCheckpointBoundaries += plan.NumCheckpointBoundaries
		mp.CombinedPeakBytes += plan.PeakBytes
		mp.CombinedPeakBeforeReuse += plan.PeakBeforeReuse
		mp.CombinedTotalBytes += plan.TotalBytes

		for _, group := range plan.AliasGroups {
			if group.SharedKey == nil || group.IsDynamic {
				continue
			}
			if group.Reason != AliasExternalBuffer && group.Reason != AliasSharedConstant {
				continue
			}
			if seenShared[group.SharedKey] {
				// Already counted via another graph: remove the duplicate from
				// every combined figure this group had contributed to.
				mp.SharedConstantDedupBytes += group.Bytes
				mp.CombinedPeakBytes -= group.Bytes
				mp.CombinedPeakBeforeReuse -= group.Bytes
				mp.CombinedTotalBytes -= group.Bytes
				continue
			}
			seenShared[group.SharedKey] = true
		}
	}
	return mp
}

// Report renders the aggregate view as deterministic text using only shapes,
// scope paths and byte counts.
func (mp *MultiPlan) Report() string {
	return formatMultiReport(mp)
}
