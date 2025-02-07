// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Mish module.
        /// </summary>
        public class Mish : torch.nn.Module
        {
            internal Mish(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Mish_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Mish_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Mish).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Mish_ctor(out IntPtr pBoxedModule);

            /// <summary>
            /// A Self Regularized Non-Monotonic Neural Activation Function.
            /// </summary>
            /// <returns></returns>
            static public Mish Mish()
            {
                var handle = THSNN_Mish_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Mish(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// A Self Regularized Non-Monotonic Neural Activation Function.
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                static public Tensor Mish(Tensor x)
                {
                    using (var m = nn.Mish()) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
