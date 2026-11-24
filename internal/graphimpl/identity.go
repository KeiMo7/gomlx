// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

import "reflect"

// bufferIdentity returns a stable identity token for a backend value. Pointer-like
// values are identified by their pointer (so the same external buffer or shared
// backend constant collapses to one identity across nodes and graphs); value-typed
// backends fall back to the value itself, which still correctly de-duplicates
// identical shared handles.
//
// It never exposes addresses in error messages: the token is only used as an
// unordered map key during build-time alias grouping.
func bufferIdentity(v any) any {
	if v == nil {
		return nil
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Ptr, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func, reflect.UnsafePointer:
		return rv.Pointer()
	default:
		return v
	}
}
