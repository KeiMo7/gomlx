// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package memory implements build-time tensor memory planning for computation graphs.
//
// A Plan is an immutable snapshot derived once from a graph while it is still being
// built: it assigns every materialized tensor output a liveness interval, groups
// tensors provably sharing the same underlying storage into alias groups (counted
// once and never assigned to a reuse slot), and then runs a conservative slot-reuse
// allocation over the remaining tensors.
//
// All figures (intervals, bytes, alias groups, peaks, slots) come from the same
// planning pass and therefore share a single accounting basis. Planning never
// mutates the graph, never starts background tasks, and takes no new dependencies.
package memory

import (
	"slices"

	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/internal/graphimpl"
	"github.com/gomlx/gomlx/support/sets"
)

// DefaultAlignment is the conservative storage alignment (in bytes) assumed for
// every tensor when the backend does not publish a stronger requirement. Two
// tensors may share a slot only when their alignment requirements match.
const DefaultAlignment = 16

// kind classifies a tensor record for reporting and liveness rules.
type kind int

const (
	kindIntermediate kind = iota
	kindParameter
	kindConstant
	kindOutput
)

// TensorRecord is one materialized tensor output observed during planning.
//
// A multi-output node contributes one record per output; each output is accounted
// separately. Records belonging to the same alias group share storage and are
// counted only once towards the peak.
type TensorRecord struct {
	// NodeId is the id of the node producing the tensor.
	NodeId graphimpl.NodeId

	// OutputIndex is the output slot within a multi-output node (0 for ordinary nodes).
	OutputIndex int

	// ScopePath is the slash-separated function scope that owns the producing node.
	ScopePath string

	// Shape of the tensor output. It may carry dynamic dimensions.
	Shape shapes.Shape

	// Bytes is the exact storage size in bytes for static shapes. It uses the
	// packed-dtype convention (sub-byte elements are rounded up to whole bytes).
	// For dynamic shapes Bytes is 0 and KnownLowerBoundBytes carries the bound.
	Bytes int64

	// KnownLowerBoundBytes is the storage lower bound computed from the static
	// dimensions of a dynamic shape (dynamic axes contribute 1 element each).
	// It equals Bytes for fully static shapes.
	KnownLowerBoundBytes int64

	// IsDynamic reports whether any dimension is only known at execution time.
	IsDynamic bool

	// Alignment required by the tensor, in bytes.
	Alignment int64

	// BornAt is the scheduling step at which the tensor is produced.
	BornAt int

	// LastUseAt is the scheduling step at which the tensor is consumed for the
	// last time. Tensors that survive the whole graph (outputs, parameters,
	// constants) have LastUseAt set to the last scheduling step.
	LastUseAt int

	// AliasGroup is the id of the alias group the tensor belongs to, or -1.
	AliasGroup int

	// Slot is the reuse slot assigned to the tensor, or -1 when the tensor is
	// part of an alias group or carries runtime dimensions.
	Slot int

	// kind is the tensor classification (not exposed beyond the package).
	kind kind
}

// IsPersistent reports whether the tensor must stay alive for the whole graph
// (graph output, parameter, or constant).
func (r *TensorRecord) IsPersistent() bool {
	return r.kind != kindIntermediate
}

// AliasGroupRecord describes a set of tensors proven to alias the same storage.
type AliasGroupRecord struct {
	// Id is the alias group id (matches TensorRecord.AliasGroup).
	Id int

	// Reason explains why the group was formed.
	Reason AliasReason

	// Members lists the (NodeId, OutputIndex) pairs in the group.
	Members []TensorRef

	// Bytes counted once for the whole group (0 when dynamic).
	Bytes int64

	// KnownLowerBoundBytes is the lower-bound size (dynamic groups).
	KnownLowerBoundBytes int64

	// IsDynamic reports whether the shared shape has runtime dimensions.
	IsDynamic bool

	// SharedKey identifies storage shared across graphs (external buffers and
	// backend-shared constants). It is nil for graph-local groups and is never
	// rendered in reports or error messages.
	SharedKey any
}

// AliasReason explains why tensors were placed in the same alias group.
type AliasReason string

const (
	// AliasBitcast: same-width reinterpretation (Bitcast with identical byte size).
	AliasBitcast AliasReason = "bitcast"

	// AliasExternalBuffer: constant taking over an externally provided buffer.
	AliasExternalBuffer AliasReason = "external_buffer"

	// AliasSharedConstant: constant reusing a backend-shared constant value,
	// including the same constant tensor used across graphs.
	AliasSharedConstant AliasReason = "shared_constant"

	// AliasRepeatedArgument: the same tensor passed to multiple function/entry inputs.
	AliasRepeatedArgument AliasReason = "repeated_argument"
)

// TensorRef identifies one tensor output within a graph.
type TensorRef struct {
	NodeId      graphimpl.NodeId
	OutputIndex int
}

// SlotReuse records one reuse-slot placement: the later tensor reuses the storage
// of an earlier, already dead tensor of identical byte size and alignment.
type SlotReuse struct {
	// Slot is the slot id (shared by both tensors).
	Slot int

	// Holder is the tensor that first occupied the slot.
	Holder TensorRef

	// Reuser is the tensor that later reused the slot.
	Reuser TensorRef

	// Bytes saved per reuse placement (the tensor's own byte size).
	Bytes int64
}

// ScopeUsage aggregates byte usage attributed to one function scope.
type ScopeUsage struct {
	ScopePath       string
	Bytes           int64
	LowerBoundBytes int64
	Dynamic         bool
}

// Plan is an immutable build-time memory planning snapshot.
//
// All fields are filled by a single planning pass and must not be mutated after
// the Plan is returned. Concurrent reads are safe because the contents are
// read-only.
type Plan struct {
	// Tensors lists every materialized tensor output, in (NodeId, OutputIndex) order.
	Tensors []*TensorRecord

	// AliasGroups lists alias groups in id order.
	AliasGroups []*AliasGroupRecord

	// Reuses lists the accepted slot-reuse placements in assignment order.
	Reuses []SlotReuse

	// ScopeUsage ranks scopes by their attributed static bytes (descending).
	ScopeUsage []ScopeUsage

	// TotalBytes is the sum of bytes of every static tensor output before
	// alias de-duplication and slot reuse.
	TotalBytes int64

	// PeakBytes is the reuse-aware storage high-water mark: each reuse slot
	// contributes one buffer, counted while one of its members is live.
	PeakBytes int64

	// PeakBeforeReuse is the conservative high-water mark when every
	// non-aliased static tensor keeps a dedicated, never-recycled buffer. It
	// is always greater than or equal to PeakBytes.
	PeakBeforeReuse int64

	// LowerBoundPeakBytes is the peak contribution of dynamic tensors evaluated
	// with their known lower bounds, combined with exact static tensors. It is
	// never mixed into PeakBytes.
	LowerBoundPeakBytes int64

	// ReusedBytes is PeakBeforeReuse - PeakBytes (bytes saved by slot reuse).
	ReusedBytes int64

	// NumDynamic is the number of tensor outputs carrying runtime dimensions.
	NumDynamic int

	// NumCheckpointBoundaries counts checkpoint boundary nodes visible in the graph.
	NumCheckpointBoundaries int

	// AliasConflicts lists alias groups whose members disagreed on static byte
	// size. Such groups are rejected: their members are conservatively treated
	// as independent tensors (each counted separately), so the plan never
	// presents a layout that only looks correct.
	AliasConflicts []AliasConflict

	// lastStep is the maximum scheduling step (for interval construction).
	lastStep int

	// refIndex maps (NodeId, OutputIndex) to its record.
	refIndex map[TensorRef]*TensorRecord
}

// AliasConflict describes an alias group rejected because its members could not
// provably share one storage allocation.
type AliasConflict struct {
	// Reason is the alias mechanism that produced the group.
	Reason AliasReason

	// Members are the tensor refs that were in the rejected group.
	Members []TensorRef

	// ExpectedBytes and ActualBytes are the disagreeing static sizes.
	ExpectedBytes int64
	ActualBytes   int64
}

// Record returns the tensor record for a reference, or nil.
func (p *Plan) Record(ref TensorRef) *TensorRecord {
	return p.refIndex[ref]
}

// NumSlots returns the number of distinct reuse slots allocated.
func (p *Plan) NumSlots() int {
	slots := sets.Make[int]()
	for _, r := range p.Tensors {
		if r.Slot >= 0 {
			slots.Insert(r.Slot)
		}
	}
	return len(slots)
}

// Options controls a planning pass.
type Options struct {
	// BudgetBytes, when > 0, enables the hard device budget: planning fails if
	// the exact static peak exceeds it. Zero means no budget is enforced.
	BudgetBytes int64

	// Outputs are the graph outputs explicitly supplied by the caller. Output
	// tensors are considered live across the whole graph. When nil, the nodes
	// recorded by the main function's Return are used.
	Outputs []*graphimpl.NodeImpl
}

// aliasKey identifies storage provenance for alias grouping.
type aliasKey struct {
	kind   AliasReason
	origin any
}

// planner holds the mutable state of one planning pass.
type planner struct {
	g              *graphimpl.GraphImpl
	opts           Options
	records        []*TensorRecord
	groupByKey     map[aliasKey]int
	groups         []*AliasGroupRecord
	pendingReuses  []SlotReuse
	outputNodeIDs  sets.Set[graphimpl.NodeId]
	aliasConflicts []AliasConflict
}

// Build runs a full planning pass over g and returns its immutable Plan.
//
// Planning is read-only and works both on a graph still being built and on a
// compiled graph whose nodes are still retained (before Finalize), so executor
// cache snapshots can be planned after compilation. Build never mutates g.
func Build(g *graphimpl.GraphImpl, opts Options) (*Plan, error) {
	if g == nil {
		return nil, newPlanError("cannot plan a nil graph")
	}
	if err := g.CheckValid(); err != nil {
		return nil, newPlanError("%s", err.Error())
	}

	p := &planner{
		g:             g,
		opts:          opts,
		groupByKey:    make(map[aliasKey]int),
		outputNodeIDs: sets.Make[graphimpl.NodeId](),
	}
	p.resolveOutputs()
	if err := p.collectRecords(); err != nil {
		return nil, err
	}
	p.computeIntervals()
	if err := p.allocateSlots(); err != nil {
		return nil, err
	}
	plan := p.summarize()
	if opts.BudgetBytes > 0 {
		if err := plan.checkBudget(opts.BudgetBytes); err != nil {
			return plan, err
		}
	}
	return plan, nil
}

// resolveOutputs collects the ids of graph outputs from explicit options or
// from recorded function returns.
func (p *planner) resolveOutputs() {
	if len(p.opts.Outputs) > 0 {
		for _, n := range p.opts.Outputs {
			if n != nil {
				p.outputNodeIDs.Insert(n.Id())
			}
		}
		return
	}
	for _, fn := range p.g.Functions() {
		for _, n := range fn.OutputNodes() {
			if n != nil {
				p.outputNodeIDs.Insert(n.Id())
			}
		}
	}
}

// collectRecords walks every node output and builds one TensorRecord per output,
// grouping aliased outputs as it goes.
func (p *planner) collectRecords() error {
	for _, node := range p.g.Nodes() {
		if node.NumOutputs() == 0 {
			continue
		}
		scopePath := ""
		if fn := node.Function(); fn != nil {
			scopePath = fn.Path()
		}
		for outIdx, shape := range node.OutputShapes() {
			if !shape.Ok() || shape.IsTuple() {
				// Tuple shapes do not correspond to a materialized leaf buffer.
				continue
			}
			rec := &TensorRecord{
				NodeId:      node.Id(),
				OutputIndex: outIdx,
				ScopePath:   scopePath,
				Shape:       shape,
				Alignment:   alignmentFor(shape),
				AliasGroup:  -1,
				Slot:        -1,
				BornAt:      int(node.Id()),
			}
			fillByteCounts(rec, shape)
			p.classify(node, rec)
			p.records = append(p.records, rec)
		}
	}
	p.buildAliasGroups()
	return nil
}

// classify marks the record kind; outputs are resolved during interval computation.
func (p *planner) classify(node *graphimpl.NodeImpl, rec *TensorRecord) {
	switch node.Type() {
	case graphimpl.NodeTypeParameter:
		// Only main-function parameters are graph inputs alive for the whole
		// graph; closure/callee parameters follow ordinary liveness intervals.
		if fn := node.Function(); fn != nil && fn.IsMain() {
			rec.kind = kindParameter
		} else {
			rec.kind = kindIntermediate
		}
	case graphimpl.NodeTypeConstant:
		rec.kind = kindConstant
	default:
		rec.kind = kindIntermediate
	}
}

// fillByteCounts computes exact bytes and the dynamic lower bound using one
// element per dynamic axis as the conservative bound.
func fillByteCounts(rec *TensorRecord, shape shapes.Shape) {
	if shape.IsDynamic() {
		rec.IsDynamic = true
		rec.Bytes = 0
		rec.KnownLowerBoundBytes = lowerBoundBytes(shape)
		return
	}
	rec.Bytes = shape.ByteSize()
	rec.KnownLowerBoundBytes = rec.Bytes
}

// lowerBoundBytes replaces every dynamic axis by 1 and applies packed rounding.
func lowerBoundBytes(shape shapes.Shape) int64 {
	numElements := 1
	for _, dim := range shape.Dimensions {
		if dim == shapes.DynamicDim {
			numElements *= 1
		} else {
			numElements *= dim
		}
	}
	bits := shape.DType.Bits()
	return int64((numElements*bits + 7) / 8)
}

// alignmentFor derives a conservative alignment requirement from the dtype.
func alignmentFor(shape shapes.Shape) int64 {
	elemBytes := int64(shape.DType.Size())
	if elemBytes <= 0 {
		// Packed sub-byte types live in byte storage.
		elemBytes = 1
	}
	align := elemBytes
	if align < DefaultAlignment {
		align = DefaultAlignment
	}
	return align
}

// computeIntervals assigns [BornAt, LastUseAt] over node creation order, which is
// a valid topological scheduling order (inputs always have smaller ids).
func (p *planner) computeIntervals() {
	lastStep := 0
	if n := len(p.g.Nodes()); n > 0 {
		lastStep = n - 1
	}

	// Map every referenced input node to its maximum consumer id.
	maxConsumer := make(map[graphimpl.NodeId]int)
	for _, node := range p.g.Nodes() {
		for _, in := range node.Inputs() {
			if in == nil {
				continue
			}
			if id := int(node.Id()); id > maxConsumer[in.Id()] {
				maxConsumer[in.Id()] = id
			}
		}
	}

	for _, rec := range p.records {
		switch {
		case rec.kind == kindParameter || rec.kind == kindConstant:
			// Parameters and constants are live across the whole graph.
			rec.LastUseAt = lastStep
		case p.outputNodeIDs.Has(rec.NodeId):
			rec.kind = kindOutput
			rec.LastUseAt = lastStep
		default:
			if last, ok := maxConsumer[rec.NodeId]; ok {
				rec.LastUseAt = last
			} else {
				// Unconsumed node: interval collapses to its birth step.
				rec.LastUseAt = rec.BornAt
			}
		}
	}
}

// buildAliasGroups forms alias groups from provable storage sharing.
func (p *planner) buildAliasGroups() {
	// 1. Same-width Bitcast: the reinterpreted output aliases its operand.
	for _, node := range p.g.Nodes() {
		if node.Type() != graphimpl.NodeTypeBitcast || len(node.Inputs()) != 1 || node.NumOutputs() != 1 {
			continue
		}
		operand := node.Inputs()[0]
		if !sameWidth(operand.Shape(), node.Shape()) {
			continue
		}
		key := aliasKey{kind: AliasBitcast, origin: [2]graphimpl.NodeId{operand.Id(), node.Id()}}
		p.join(TensorRef{NodeId: operand.Id()}, TensorRef{NodeId: node.Id()}, AliasBitcast, key)
	}

	// 2 & 3. Constants backed by external buffers or shared backend values.
	for _, node := range p.g.Nodes() {
		if node.Type() != graphimpl.NodeTypeConstant {
			continue
		}
		ni, ok := node.NodeInputs().(*graphimpl.NodeInputsConstant)
		if !ok || ni.ExternalIdentity == nil {
			continue
		}
		reason := AliasSharedConstant
		if ni.Provenance == graphimpl.ConstantProvenanceExternalBuffer {
			reason = AliasExternalBuffer
		}
		key := aliasKey{kind: reason, origin: ni.ExternalIdentity}
		p.joinConstant(node, key, reason)
	}

	// 4. Repeated arguments: one node feeding multiple call inputs.
	p.collectRepeatedArguments()

	for _, group := range p.groups {
		p.finalizeGroup(group)
	}
	p.detectAliasConflicts()
}

// detectAliasConflicts rejects groups whose members disagree on static byte
// size. A rejected group is dissolved: its members fall back to independent,
// conservatively counted tensors, and the conflict is recorded on the plan so a
// caller never receives a layout that only looks correct.
func (p *planner) detectAliasConflicts() {
	var surviving []*AliasGroupRecord
	for _, group := range p.groups {
		var expectedBytes int64 = -1
		conflict := false
		for _, member := range group.Members {
			rec := p.refRecord(member)
			if rec == nil || rec.IsDynamic {
				continue
			}
			if expectedBytes == -1 {
				expectedBytes = rec.Bytes
			} else if rec.Bytes != expectedBytes {
				conflict = true
				p.aliasConflicts = append(p.aliasConflicts, AliasConflict{
					Reason:        group.Reason,
					Members:       slices.Clone(group.Members),
					ExpectedBytes: expectedBytes,
					ActualBytes:   rec.Bytes,
				})
				break
			}
		}
		if conflict {
			// Dissolve the group: every member becomes an independent tensor.
			for _, member := range group.Members {
				if rec := p.refRecord(member); rec != nil {
					rec.AliasGroup = -1
				}
			}
			continue
		}
		surviving = append(surviving, group)
	}
	p.groups = surviving
}

// sameWidth reports whether two shapes occupy identical storage bytes.
func sameWidth(a, b shapes.Shape) bool {
	if a.IsDynamic() || b.IsDynamic() {
		return false
	}
	return a.ByteSize() == b.ByteSize() && a.DType.Bits() == b.DType.Bits()
}

// join merges two refs into the same alias group, creating the group on first use.
func (p *planner) join(a, b TensorRef, reason AliasReason, key aliasKey) {
	groupID, found := p.groupByKey[key]
	if !found {
		groupID = len(p.groups)
		p.groups = append(p.groups, &AliasGroupRecord{Id: groupID, Reason: reason})
		p.groupByKey[key] = groupID
	}
	p.addToGroup(groupID, a)
	p.addToGroup(groupID, b)
}

// joinConstant joins a constant node's outputs into the group for its provenance.
func (p *planner) joinConstant(node *graphimpl.NodeImpl, key aliasKey, reason AliasReason) {
	groupID, found := p.groupByKey[key]
	if !found {
		groupID = len(p.groups)
		p.groups = append(p.groups, &AliasGroupRecord{Id: groupID, Reason: reason, SharedKey: key.origin})
		p.groupByKey[key] = groupID
	}
	for outIdx := range node.NumOutputs() {
		p.addToGroup(groupID, TensorRef{NodeId: node.Id(), OutputIndex: outIdx})
	}
}

// addToGroup inserts a ref if not already present.
func (p *planner) addToGroup(groupID int, ref TensorRef) {
	group := p.groups[groupID]
	for _, m := range group.Members {
		if m == ref {
			return
		}
	}
	group.Members = append(group.Members, ref)
}

// collectRepeatedArguments finds nodes where one underlying tensor feeds more
// than one input edge. For ordinary ops this records the shared source; for
// function calls it also pulls in the callee's entry parameters, which are
// distinct nodes aliasing the caller's storage at runtime.
func (p *planner) collectRepeatedArguments() {
	for _, node := range p.g.Nodes() {
		positions := make(map[graphimpl.NodeId][]int)
		for idx, in := range node.Inputs() {
			if in != nil {
				positions[in.Id()] = append(positions[in.Id()], idx)
			}
		}
		var calleeParams []*graphimpl.NodeImpl
		if call, ok := node.NodeInputs().(*graphimpl.NodeInputsCall); ok && call.F != nil {
			calleeParams = call.F.Parameters()
		}
		for inputID, pos := range positions {
			if len(pos) < 2 {
				continue
			}
			key := aliasKey{kind: AliasRepeatedArgument, origin: [2]graphimpl.NodeId{node.Id(), inputID}}
			groupID, found := p.groupByKey[key]
			if !found {
				groupID = len(p.groups)
				p.groups = append(p.groups, &AliasGroupRecord{Id: groupID, Reason: AliasRepeatedArgument})
				p.groupByKey[key] = groupID
			}
			p.addToGroup(groupID, TensorRef{NodeId: inputID})
			// Callee entry parameters receiving the same argument alias it.
			for _, idx := range pos {
				if idx < len(calleeParams) && calleeParams[idx] != nil {
					p.addToGroup(groupID, TensorRef{NodeId: calleeParams[idx].Id()})
				}
			}
		}
	}
}

// finalizeGroup computes group-level byte accounting from its first member and
// stamps the group id onto every member record.
func (p *planner) finalizeGroup(group *AliasGroupRecord) {
	if len(group.Members) == 0 {
		return
	}
	first := p.refRecord(group.Members[0])
	if first == nil {
		return
	}
	group.Bytes = first.Bytes
	group.KnownLowerBoundBytes = first.KnownLowerBoundBytes
	group.IsDynamic = first.IsDynamic
	for _, member := range group.Members {
		if rec := p.refRecord(member); rec != nil {
			rec.AliasGroup = group.Id
		}
	}
}

// refRecord finds a collected record by reference.
func (p *planner) refRecord(ref TensorRef) *TensorRecord {
	for _, rec := range p.records {
		if rec.NodeId == ref.NodeId && rec.OutputIndex == ref.OutputIndex {
			return rec
		}
	}
	return nil
}

// nodeByID returns a node by id without asserting the building state, so the
// read-only planner also works on a compiled graph whose nodes are retained.
func (p *planner) nodeByID(id graphimpl.NodeId) *graphimpl.NodeImpl {
	nodes := p.g.Nodes()
	if id < 0 || int(id) >= len(nodes) {
		return nil
	}
	return nodes[id]
}

// allocateSlots runs conservative first-fit slot reuse: two tensors share a slot
// only when their intervals do not overlap, their exact bytes match, and their
// alignment matches. Alias-group tensors and dynamic tensors never enter the pool.
func (p *planner) allocateSlots() error {
	type slotState struct {
		id        int
		bytes     int64
		alignment int64
		freeAt    int
		holder    TensorRef
	}
	var slots []*slotState

	ordered := slices.Clone(p.records)
	slices.SortFunc(ordered, func(a, b *TensorRecord) int {
		if a.BornAt != b.BornAt {
			return a.BornAt - b.BornAt
		}
		if a.NodeId != b.NodeId {
			return int(a.NodeId - b.NodeId)
		}
		return a.OutputIndex - b.OutputIndex
	})

	for _, rec := range ordered {
		if rec.AliasGroup >= 0 || rec.IsDynamic {
			continue
		}
		placed := false
		for _, slot := range slots {
			if slot.bytes != rec.Bytes || slot.alignment != rec.Alignment {
				continue
			}
			if slot.freeAt >= rec.BornAt {
				continue // intervals overlap: conservative refusal
			}
			rec.Slot = slot.id
			ref := TensorRef{NodeId: rec.NodeId, OutputIndex: rec.OutputIndex}
			p.pendingReuses = append(p.pendingReuses, SlotReuse{
				Slot: slot.id, Holder: slot.holder, Reuser: ref, Bytes: rec.Bytes,
			})
			slot.freeAt = rec.LastUseAt
			slot.holder = ref
			placed = true
			break
		}
		if !placed {
			ref := TensorRef{NodeId: rec.NodeId, OutputIndex: rec.OutputIndex}
			slots = append(slots, &slotState{
				id: len(slots), bytes: rec.Bytes, alignment: rec.Alignment,
				freeAt: rec.LastUseAt, holder: ref,
			})
			rec.Slot = len(slots) - 1
		}
	}
	return nil
}

// summarize builds the immutable Plan with totals, peaks and scope ranking.
func (p *planner) summarize() *Plan {
	plan := &Plan{
		Tensors:        p.records,
		AliasGroups:    p.groups,
		AliasConflicts: p.aliasConflicts,
		Reuses:         p.pendingReuses,
		lastStep:       len(p.g.Nodes()) - 1,
		refIndex:       make(map[TensorRef]*TensorRecord, len(p.records)),
	}
	for _, rec := range p.records {
		plan.refIndex[TensorRef{NodeId: rec.NodeId, OutputIndex: rec.OutputIndex}] = rec
		if rec.IsDynamic {
			plan.NumDynamic++
		} else {
			plan.TotalBytes += rec.Bytes
		}
		if node := p.nodeByID(rec.NodeId); node != nil && node.IsCheckpointed() && rec.OutputIndex == 0 {
			plan.NumCheckpointBoundaries++
		}
	}

	plan.PeakBeforeReuse = p.simulatePeak(false)
	plan.PeakBytes = p.simulatePeak(true)
	plan.ReusedBytes = plan.PeakBeforeReuse - plan.PeakBytes
	plan.LowerBoundPeakBytes = p.simulateLowerBoundPeak()
	plan.ScopeUsage = p.rankScopes()
	return plan
}

// simulatePeak computes the storage high-water mark under two allocation
// conventions, sharing the same interval/byte/alias basis.
//
// With reuse disabled, every non-aliased static tensor receives a dedicated
// buffer that is never recycled: the high-water mark at a step counts every
// tensor already born (its buffer stays reserved even after the tensor dies).
// This is the conservative "one buffer per tensor" figure a reviewer can audit
// without trusting the reuse allocator.
//
// With reuse enabled, tensors assigned to the same slot share one physical
// buffer and a dead tensor releases its slot for a later same-sized reuser. The
// high-water mark counts each slot's buffer once when one of its members is
// live. This is the figure an actual reuse-aware allocator reaches, and it is
// always lower than or equal to the no-reuse mark.
func (p *planner) simulatePeak(withReuse bool) int64 {
	groupLive := make(map[int]bool)
	var peak int64

	// Precompute, per slot, the tensors assigned to it.
	slotTensors := make(map[int][]*TensorRecord)
	for _, rec := range p.records {
		if rec.IsDynamic || rec.AliasGroup >= 0 || rec.Slot < 0 {
			continue
		}
		slotTensors[rec.Slot] = append(slotTensors[rec.Slot], rec)
	}

	for step := 0; step <= p.lastStep(); step++ {
		for gid := range groupLive {
			group := p.groups[gid]
			alive := false
			for _, m := range group.Members {
				if rec := p.refRecord(m); rec != nil && rec.LastUseAt >= step && rec.BornAt <= step {
					alive = true
					break
				}
			}
			if !alive {
				delete(groupLive, gid)
			}
		}
		var total int64
		if withReuse {
			// Reuse-aware: one physical buffer per slot, counted once while a
			// member is live. A released slot frees its buffer for a reuser.
			for _, members := range slotTensors {
				for _, rec := range members {
					if rec.BornAt <= step && rec.LastUseAt >= step {
						total += rec.Bytes // identical slot size for every member
						break
					}
				}
			}
		} else {
			// Conservative: a dedicated, never-recycled buffer per tensor,
			// counted from its birth step to the end of the graph.
			for _, rec := range p.records {
				if rec.IsDynamic || rec.AliasGroup >= 0 {
					continue
				}
				if rec.BornAt <= step {
					total += rec.Bytes
				}
			}
		}

		for _, rec := range p.records {
			if rec.BornAt != step || rec.IsDynamic || rec.AliasGroup < 0 {
				continue
			}
			groupLive[rec.AliasGroup] = true
		}
		for gid := range groupLive {
			total += p.groups[gid].Bytes
		}
		if total > peak {
			peak = total
		}
	}
	return peak
}

// lastStep returns the maximum node id, or -1 for an empty graph.
func (p *planner) lastStep() int {
	if count := len(p.g.Nodes()); count == 0 {
		return -1
	} else {
		return count - 1
	}
}

// simulateLowerBoundPeak combines exact static tensors with dynamic tensors
// evaluated at their known lower bounds; dynamic figures stay separate from the
// exact peak.
func (p *planner) simulateLowerBoundPeak() int64 {
	live := make(map[TensorRef]int64)
	groupLive := make(map[int]int64)
	var peak int64

	for step := 0; step <= p.lastStep(); step++ {
		for ref := range live {
			if rec := p.refRecord(ref); rec != nil && rec.LastUseAt < step {
				delete(live, ref)
			}
		}
		for gid := range groupLive {
			group := p.groups[gid]
			alive := false
			for _, m := range group.Members {
				if rec := p.refRecord(m); rec != nil && rec.LastUseAt >= step && rec.BornAt <= step {
					alive = true
					break
				}
			}
			if !alive {
				delete(groupLive, gid)
			}
		}

		for _, rec := range p.records {
			if rec.BornAt != step {
				continue
			}
			ref := TensorRef{NodeId: rec.NodeId, OutputIndex: rec.OutputIndex}
			if rec.AliasGroup >= 0 {
				groupLive[rec.AliasGroup] = rec.KnownLowerBoundBytes
				continue
			}
			if _, ok := live[ref]; ok {
				continue
			}
			live[ref] = rec.KnownLowerBoundBytes
		}

		var total int64
		for _, b := range live {
			total += b
		}
		for _, b := range groupLive {
			total += b
		}
		if total > peak {
			peak = total
		}
	}
	return peak
}

// rankScopes aggregates per-scope byte usage and sorts descending.
func (p *planner) rankScopes() []ScopeUsage {
	type acc struct {
		bytes, lower int64
		dynamic      bool
	}
	byScope := make(map[string]*acc)
	for _, rec := range p.records {
		a := byScope[rec.ScopePath]
		if a == nil {
			a = &acc{}
			byScope[rec.ScopePath] = a
		}
		if rec.IsDynamic {
			a.lower += rec.KnownLowerBoundBytes
			a.dynamic = true
		} else if rec.AliasGroup < 0 {
			a.bytes += rec.Bytes
		}
	}
	out := make([]ScopeUsage, 0, len(byScope))
	for path, a := range byScope {
		out = append(out, ScopeUsage{ScopePath: path, Bytes: a.bytes, LowerBoundBytes: a.lower, Dynamic: a.dynamic})
	}
	slices.SortFunc(out, func(a, b ScopeUsage) int {
		if a.Bytes != b.Bytes {
			return int(b.Bytes - a.Bytes)
		}
		if a.LowerBoundBytes != b.LowerBoundBytes {
			return int(b.LowerBoundBytes - a.LowerBoundBytes)
		}
		return compareStrings(a.ScopePath, b.ScopePath)
	})
	return out
}

// checkBudget enforces the hard device budget against exact figures first, then
// the known lower bound. Dynamic uncertainty never silently passes a budget.
func (p *Plan) checkBudget(budget int64) error {
	if p.PeakBytes > budget {
		return newBudgetExceededError(p, budget, p.PeakBytes, false)
	}
	if p.LowerBoundPeakBytes > budget {
		return newBudgetExceededError(p, budget, p.LowerBoundPeakBytes, true)
	}
	return nil
}

// compareStrings is a deterministic string comparator.
func compareStrings(a, b string) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}
