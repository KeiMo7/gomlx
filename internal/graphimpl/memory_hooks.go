// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

// MemoryPlanResult is the opaque result returned by a registered build-time
// memory planner. The concrete type lives in internal/graphimpl/memory; graphimpl
// only forwards it so the executor can enforce a budget without an import cycle.
type MemoryPlanResult any

// MemoryPlannerHook runs a build-time memory planning pass with an optional hard
// budget (0 disables the budget) and explicit graph outputs.
type MemoryPlannerHook func(g *GraphImpl, budgetBytes int64, outputs []*NodeImpl) (MemoryPlanResult, error)

// MemoryRematerializeHook inserts framework-chosen checkpoint boundaries until
// the planned peak fits budgetBytes and returns the resulting plan.
type MemoryRematerializeHook func(g *GraphImpl, budgetBytes int64, outputs []*NodeImpl) (MemoryPlanResult, []NodeId, error)

// Registered memory-planning hooks, set by the public core/graph/memory package
// in its init(). Both are nil when that package is not imported, preserving the
// default compile/execute path exactly.
var (
	memoryPlannerHook       MemoryPlannerHook
	memoryRematerializeHook MemoryRematerializeHook
)

// RegisterMemoryPlanner installs the build-time planner hooks. It is intended to
// be called once from an init() function.
func RegisterMemoryPlanner(planner MemoryPlannerHook, remat MemoryRematerializeHook) {
	memoryPlannerHook = planner
	memoryRematerializeHook = remat
}

// MemoryPlannerInstalled reports whether a build-time planner was registered.
func MemoryPlannerInstalled() bool {
	return memoryPlannerHook != nil
}

// RunMemoryPlanner invokes the registered planner hook.
func RunMemoryPlanner(g *GraphImpl, budgetBytes int64, outputs []*NodeImpl) (MemoryPlanResult, error) {
	return memoryPlannerHook(g, budgetBytes, outputs)
}

// RunMemoryRematerialize invokes the registered rematerialization hook.
func RunMemoryRematerialize(g *GraphImpl, budgetBytes int64, outputs []*NodeImpl) (MemoryPlanResult, []NodeId, error) {
	return memoryRematerializeHook(g, budgetBytes, outputs)
}
