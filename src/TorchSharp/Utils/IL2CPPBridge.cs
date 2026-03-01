// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
//
// Static callback infrastructure for IL2CPP (Unity AOT) compatibility.
// IL2CPP cannot marshal delegates that point to instance methods or closures
// to native code.  This file provides static-method alternatives that use
// slot pools, thread-static storage, and registries to dispatch to the
// correct managed instance.

using System;
using System.Collections.Concurrent;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    /// <summary>
    /// MonoPInvokeCallback attribute for IL2CPP / AOT reverse-P/Invoke support.
    /// IL2CPP recognises this attribute by name to generate native→managed trampolines.
    /// Defined here because netstandard2.0 does not ship it.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    internal sealed class MonoPInvokeCallbackAttribute : Attribute
    {
        public MonoPInvokeCallbackAttribute(Type delegateType) { }
    }

    internal static class IL2CPPBridge
    {
        // ═══════════════════════════════════════════════════════════════════
        //  PinnedArray Allocator Slot Pool
        // ═══════════════════════════════════════════════════════════════════
        //  All PinnedArray allocator callbacks are invoked synchronously
        //  (inline inside the P/Invoke call).  Each PinnedArray instance
        //  acquires a slot from this pool on first delegate use and releases
        //  it on Dispose().

        internal const int MaxAllocSlots = 16;
        private static readonly object s_allocLock = new object();
        private static readonly bool[] s_allocUsed = new bool[MaxAllocSlots];
        internal static readonly Func<IntPtr, IntPtr>[] s_allocFn = new Func<IntPtr, IntPtr>[MaxAllocSlots];

        internal static int AcquireAllocSlot(Func<IntPtr, IntPtr> fn)
        {
            lock (s_allocLock)
            {
                for (int i = 0; i < MaxAllocSlots; i++)
                {
                    if (!s_allocUsed[i])
                    {
                        s_allocUsed[i] = true;
                        s_allocFn[i] = fn;
                        return i;
                    }
                }
            }
            throw new InvalidOperationException(
                $"PinnedArray allocator slot pool exhausted ({MaxAllocSlots}). " +
                "Ensure PinnedArray instances are Disposed promptly.");
        }

        internal static void ReleaseAllocSlot(int slot)
        {
            if (slot < 0) return;
            lock (s_allocLock)
            {
                s_allocUsed[slot] = false;
                s_allocFn[slot] = null;
            }
        }

        // 16 pre-defined static callbacks — each dispatches to its slot's Func
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A0(IntPtr l) => s_allocFn[0](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A1(IntPtr l) => s_allocFn[1](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A2(IntPtr l) => s_allocFn[2](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A3(IntPtr l) => s_allocFn[3](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A4(IntPtr l) => s_allocFn[4](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A5(IntPtr l) => s_allocFn[5](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A6(IntPtr l) => s_allocFn[6](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A7(IntPtr l) => s_allocFn[7](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A8(IntPtr l) => s_allocFn[8](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A9(IntPtr l) => s_allocFn[9](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A10(IntPtr l) => s_allocFn[10](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A11(IntPtr l) => s_allocFn[11](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A12(IntPtr l) => s_allocFn[12](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A13(IntPtr l) => s_allocFn[13](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A14(IntPtr l) => s_allocFn[14](l);
        [MonoPInvokeCallback(typeof(AllocatePinnedArray))] private static IntPtr A15(IntPtr l) => s_allocFn[15](l);

        internal static readonly AllocatePinnedArray[] AllocDelegates =
        {
            A0, A1, A2, A3, A4, A5, A6, A7,
            A8, A9, A10, A11, A12, A13, A14, A15,
        };

        // ═══════════════════════════════════════════════════════════════════
        //  Custom Module Forward Trampolines
        // ═══════════════════════════════════════════════════════════════════
        //  Forward callbacks are stored by native CustomModule and invoked
        //  later (not synchronously), so we need per-module static methods.

        internal const int MaxFwdSlots = 8;
        internal static readonly torch.nn.Module[] s_fwdModules = new torch.nn.Module[MaxFwdSlots];
        private static int s_nextFwd;

        internal static int AcquireFwdSlot(torch.nn.Module module)
        {
            int slot = System.Threading.Interlocked.Increment(ref s_nextFwd) - 1;
            if (slot >= MaxFwdSlots)
                throw new InvalidOperationException(
                    $"Custom nn.Module forward slot pool exhausted ({MaxFwdSlots} max).");
            s_fwdModules[slot] = module;
            return slot;
        }

        private static IntPtr FwdDispatch(int slot, IntPtr t)
        {
            var m = s_fwdModules[slot];
            if (m == null) return t;
            var input = new torch.Tensor(t);
            var output = ((torch.nn.Module<torch.Tensor, torch.Tensor>)m).call(input);
            input.DecoupleFromNativeHandle();
            return output.DecoupleFromNativeHandle();
        }

        [MonoPInvokeCallback(typeof(ForwardFunctionC))] private static IntPtr F0(IntPtr t) => FwdDispatch(0, t);
        [MonoPInvokeCallback(typeof(ForwardFunctionC))] private static IntPtr F1(IntPtr t) => FwdDispatch(1, t);
        [MonoPInvokeCallback(typeof(ForwardFunctionC))] private static IntPtr F2(IntPtr t) => FwdDispatch(2, t);
        [MonoPInvokeCallback(typeof(ForwardFunctionC))] private static IntPtr F3(IntPtr t) => FwdDispatch(3, t);
        [MonoPInvokeCallback(typeof(ForwardFunctionC))] private static IntPtr F4(IntPtr t) => FwdDispatch(4, t);
        [MonoPInvokeCallback(typeof(ForwardFunctionC))] private static IntPtr F5(IntPtr t) => FwdDispatch(5, t);
        [MonoPInvokeCallback(typeof(ForwardFunctionC))] private static IntPtr F6(IntPtr t) => FwdDispatch(6, t);
        [MonoPInvokeCallback(typeof(ForwardFunctionC))] private static IntPtr F7(IntPtr t) => FwdDispatch(7, t);

        internal static readonly ForwardFunctionC[] FwdDelegates =
        {
            F0, F1, F2, F3, F4, F5, F6, F7,
        };

        // ═══════════════════════════════════════════════════════════════════
        //  GCHandle Deleter  (Tensor factory data-pinning cleanup)
        // ═══════════════════════════════════════════════════════════════════
        //  Maps data-pointer → cleanup action.  When native code frees the
        //  tensor storage it calls StaticDeleter(dataPtr), which looks up
        //  and invokes the corresponding cleanup.

        private static readonly ConcurrentDictionary<IntPtr, Action> s_delActions =
            new ConcurrentDictionary<IntPtr, Action>();

        internal static void RegisterDeleter(IntPtr dataPtr, Action cleanup)
        {
            s_delActions[dataPtr] = cleanup;
        }

        [MonoPInvokeCallback(typeof(PInvoke.GCHandleDeleter))]
        private static void StaticDeleter(IntPtr dataPtr)
        {
            if (s_delActions.TryRemove(dataPtr, out var action))
                action();
        }

        internal static readonly PInvoke.GCHandleDeleter DeleterDelegate = StaticDeleter;

        // ═══════════════════════════════════════════════════════════════════
        //  NativeTensorOrScalarIndexedArray Allocator Slots
        // ═══════════════════════════════════════════════════════════════════
        //  Same pool pattern as PinnedArray but for the two-parameter
        //  AllocateIndexedNativeTensorOrScalarArray delegate.

        internal const int MaxIdxAllocSlots = 8;
        private static readonly object s_idxAllocLock = new object();
        private static readonly bool[] s_idxAllocUsed = new bool[MaxIdxAllocSlots];
        internal static readonly Func<int, IntPtr, IntPtr>[] s_idxAllocFn =
            new Func<int, IntPtr, IntPtr>[MaxIdxAllocSlots];

        internal static int AcquireIdxAllocSlot(Func<int, IntPtr, IntPtr> fn)
        {
            lock (s_idxAllocLock)
            {
                for (int i = 0; i < MaxIdxAllocSlots; i++)
                {
                    if (!s_idxAllocUsed[i])
                    {
                        s_idxAllocUsed[i] = true;
                        s_idxAllocFn[i] = fn;
                        return i;
                    }
                }
            }
            throw new InvalidOperationException(
                $"IndexedArray allocator slot pool exhausted ({MaxIdxAllocSlots}).");
        }

        internal static void ReleaseIdxAllocSlot(int slot)
        {
            if (slot < 0) return;
            lock (s_idxAllocLock)
            {
                s_idxAllocUsed[slot] = false;
                s_idxAllocFn[slot] = null;
            }
        }

        [MonoPInvokeCallback(typeof(AllocateIndexedNativeTensorOrScalarArray))]
        private static IntPtr IA0(int id, IntPtr l) => s_idxAllocFn[0](id, l);
        [MonoPInvokeCallback(typeof(AllocateIndexedNativeTensorOrScalarArray))]
        private static IntPtr IA1(int id, IntPtr l) => s_idxAllocFn[1](id, l);
        [MonoPInvokeCallback(typeof(AllocateIndexedNativeTensorOrScalarArray))]
        private static IntPtr IA2(int id, IntPtr l) => s_idxAllocFn[2](id, l);
        [MonoPInvokeCallback(typeof(AllocateIndexedNativeTensorOrScalarArray))]
        private static IntPtr IA3(int id, IntPtr l) => s_idxAllocFn[3](id, l);
        [MonoPInvokeCallback(typeof(AllocateIndexedNativeTensorOrScalarArray))]
        private static IntPtr IA4(int id, IntPtr l) => s_idxAllocFn[4](id, l);
        [MonoPInvokeCallback(typeof(AllocateIndexedNativeTensorOrScalarArray))]
        private static IntPtr IA5(int id, IntPtr l) => s_idxAllocFn[5](id, l);
        [MonoPInvokeCallback(typeof(AllocateIndexedNativeTensorOrScalarArray))]
        private static IntPtr IA6(int id, IntPtr l) => s_idxAllocFn[6](id, l);
        [MonoPInvokeCallback(typeof(AllocateIndexedNativeTensorOrScalarArray))]
        private static IntPtr IA7(int id, IntPtr l) => s_idxAllocFn[7](id, l);

        internal static readonly AllocateIndexedNativeTensorOrScalarArray[] IdxAllocDelegates =
        {
            IA0, IA1, IA2, IA3, IA4, IA5, IA6, IA7,
        };
    }
}
