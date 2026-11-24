// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package memory

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute/support/humanize"
)

// formatMultiReport renders the executor-level aggregate view. It never includes
// tensor values, weight paths or external addresses — only counts and bytes.
func formatMultiReport(mp *MultiPlan) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Multi-graph memory plan (%d graphs):\n", len(mp.Plans))
	fmt.Fprintf(&b, "  combined total bytes:          %s\n", humanize.Bytes(uint64(mp.CombinedTotalBytes)))
	fmt.Fprintf(&b, "  combined peak before reuse:   %s\n", humanize.Bytes(uint64(mp.CombinedPeakBeforeReuse)))
	fmt.Fprintf(&b, "  combined peak after reuse:    %s\n", humanize.Bytes(uint64(mp.CombinedPeakBytes)))
	fmt.Fprintf(&b, "  shared-constant dedup saved:  %s\n", humanize.Bytes(uint64(mp.SharedConstantDedupBytes)))
	fmt.Fprintf(&b, "  runtime-dimension outputs:    %d\n", mp.NumDynamic)
	fmt.Fprintf(&b, "  checkpoint boundaries:        %d\n", mp.NumCheckpointBoundaries)
	return b.String()
}
