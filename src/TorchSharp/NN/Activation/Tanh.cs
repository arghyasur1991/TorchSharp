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
        /// This class is used to represent a Tanh module.
        /// </summary>
        public class Tanh : torch.nn.Module
        {
            internal Tanh(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Tanh_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Tanh_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Tanh).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Tanh_ctor(out IntPtr pBoxedModule);

            /// <summary>
            /// Tanh activation
            /// </summary>
            /// <returns></returns>
            static public Tanh Tanh()
            {
                var handle = THSNN_Tanh_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tanh(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Tanh activation
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                static public Tensor tanh(Tensor x)
                {
                    using (var m = nn.Tanh()) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
