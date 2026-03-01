// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    /// <summary>
    /// Allocator of T[] that pins the memory and handles unpinning.
    /// (taken from StackOverflow)
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal sealed class PinnedArray<T> : IDisposable where T : struct
    {
        private GCHandle handle;
        private int _callbackSlot = -1;

        public T[] Array { get; private set; }

        public IntPtr CreateArray(int length)
        {
            FreeHandle();

            Array = new T[length];

            // try... finally trick to be sure that the code isn't interrupted by asynchronous exceptions
            try {
            } finally {
                handle = GCHandle.Alloc(Array, GCHandleType.Pinned);
            }

            return handle.AddrOfPinnedObject();
        }

        public IntPtr CreateArray(IntPtr length)
        {
            return CreateArray((int)length);
        }

        public IntPtr CreateArray(T[] array)
        {
            FreeHandle();

            Array = array;

            // try... finally trick to be sure that the code isn't interrupted by asynchronous exceptions
            try {
            } finally {
                handle = GCHandle.Alloc(Array, GCHandleType.Pinned);
            }

            return handle.AddrOfPinnedObject();
        }

        /// <summary>
        /// IL2CPP-safe delegate backed by a static method.
        /// Acquires a slot from the global pool on first access;
        /// the slot is released on Dispose().
        /// </summary>
        public AllocatePinnedArray Allocator
        {
            get
            {
                if (_callbackSlot < 0)
                    _callbackSlot = IL2CPPBridge.AcquireAllocSlot(new Func<IntPtr, IntPtr>(CreateArray));
                return IL2CPPBridge.AllocDelegates[_callbackSlot];
            }
        }

        public void Dispose()
        {
            if (Array != null) {
                foreach (var val in Array) {
                    (val as IDisposable)?.Dispose();
                }
            }
            if (_callbackSlot >= 0) {
                IL2CPPBridge.ReleaseAllocSlot(_callbackSlot);
                _callbackSlot = -1;
            }
            FreeHandle();
        }

        ~PinnedArray()
        {
            if (_callbackSlot >= 0) {
                IL2CPPBridge.ReleaseAllocSlot(_callbackSlot);
                _callbackSlot = -1;
            }
            FreeHandle();
        }

        private void FreeHandle()
        {
            if (handle.IsAllocated) {
                handle.Free();
            }
        }
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate IntPtr AllocatePinnedArray(IntPtr length);
}
