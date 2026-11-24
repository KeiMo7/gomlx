// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package memory

import (
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/gomlx/compute/support/humanize"
)

// topTensors returns the largest static tensor outputs, descending by bytes.
// Dynamic tensors are omitted from the exact ranking but counted separately.
func (p *Plan) topTensors(limit int) []TopTensor {
	type ranked struct {
		rec *TensorRecord
	}
	var all []ranked
	for _, rec := range p.Tensors {
		if rec.IsDynamic || rec.AliasGroup >= 0 {
			continue
		}
		all = append(all, ranked{rec})
	}
	slices.SortFunc(all, func(a, b ranked) int {
		if a.rec.Bytes != b.rec.Bytes {
			return int(b.rec.Bytes - a.rec.Bytes)
		}
		if a.rec.NodeId != b.rec.NodeId {
			return int(a.rec.NodeId - b.rec.NodeId)
		}
		return a.rec.OutputIndex - b.rec.OutputIndex
	})
	if limit > 0 && len(all) > limit {
		all = all[:limit]
	}
	out := make([]TopTensor, 0, len(all))
	for _, item := range all {
		rec := item.rec
		out = append(out, TopTensor{
			NodeId:      int64(rec.NodeId),
			OutputIndex: rec.OutputIndex,
			ScopePath:   rec.ScopePath,
			Bytes:       rec.Bytes,
			ShapeText:   rec.Shape.String(),
		})
	}
	return out
}

// Report renders the planning snapshot as a deterministic, human-readable text
// block. It only includes shapes, scope paths and byte counts.
func (p *Plan) Report() string {
	var b strings.Builder
	p.WriteReport(&b)
	return b.String()
}

// WriteReport writes the planning snapshot to w.
func (p *Plan) WriteReport(w io.Writer) {
	fmt.Fprintf(w, "Memory plan:\n")
	fmt.Fprintf(w, "  tensor outputs:        %d (%d with runtime dimensions)\n", len(p.Tensors), p.NumDynamic)
	fmt.Fprintf(w, "  total static bytes:    %s\n", humanize.Bytes(uint64(p.TotalBytes)))
	fmt.Fprintf(w, "  peak before reuse:     %s\n", humanize.Bytes(uint64(p.PeakBeforeReuse)))
	fmt.Fprintf(w, "  peak after reuse:      %s\n", humanize.Bytes(uint64(p.PeakBytes)))
	fmt.Fprintf(w, "  bytes saved by reuse:  %s (%d reuse placements, %d slots)\n",
		humanize.Bytes(uint64(p.ReusedBytes)), len(p.Reuses), p.NumSlots())
	fmt.Fprintf(w, "  lower-bound peak:      %s\n", humanize.Bytes(uint64(p.LowerBoundPeakBytes)))
	fmt.Fprintf(w, "  alias groups:          %d\n", len(p.AliasGroups))
	if len(p.AliasConflicts) > 0 {
		fmt.Fprintf(w, "  alias conflicts:       %d (dissolved; members counted independently)\n", len(p.AliasConflicts))
	}
	fmt.Fprintf(w, "  checkpoint boundaries: %d\n", p.NumCheckpointBoundaries)

	if len(p.ScopeUsage) > 0 {
		fmt.Fprintf(w, "  scope ranking:\n")
		for _, su := range p.ScopeUsage {
			dyn := ""
			if su.Dynamic {
				dyn = fmt.Sprintf(" (+%s lower-bound dynamic)", humanize.Bytes(uint64(su.LowerBoundBytes)))
			}
			fmt.Fprintf(w, "    %-24s %s%s\n", su.ScopePath, humanize.Bytes(uint64(su.Bytes)), dyn)
		}
	}

	if len(p.AliasGroups) > 0 {
		fmt.Fprintf(w, "  alias detail:\n")
		for _, g := range p.AliasGroups {
			parts := make([]string, 0, len(g.Members))
			for _, m := range g.Members {
				parts = append(parts, fmt.Sprintf("#%d:%d", m.NodeId, m.OutputIndex))
			}
			size := humanize.Bytes(uint64(g.Bytes))
			if g.IsDynamic {
				size = fmt.Sprintf("dynamic (>= %s)", humanize.Bytes(uint64(g.KnownLowerBoundBytes)))
			}
			fmt.Fprintf(w, "    group %d [%s] %s -> %s\n", g.Id, g.Reason, strings.Join(parts, ","), size)
		}
	}

	if top := p.topTensors(5); len(top) > 0 {
		fmt.Fprintf(w, "  largest tensors:\n")
		for _, t := range top {
			fmt.Fprintf(w, "    node #%d:%d %s %s (%s)\n",
				t.NodeId, t.OutputIndex, t.ShapeText, humanize.Bytes(uint64(t.Bytes)), t.ScopePath)
		}
	}
}

// ReusedPairs returns the accepted reuse placements for external inspection.
func (p *Plan) ReusedPairs() []SlotReuse {
	return slices.Clone(p.Reuses)
}
