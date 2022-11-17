using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace NeuronalNet
{
    public class SignalVector
    {
        IntPtr vec = IntPtr.Zero;
        bool thisOwner = false;

        [DllImport(Net.neuralNetDllName)] static extern IntPtr SignalVector_instantiate1();
        [DllImport(Net.neuralNetDllName)] static extern IntPtr SignalVector_instantiate2(ulong size);
        [DllImport(Net.neuralNetDllName)] static extern IntPtr SignalVector_instantiate3(IntPtr other); //const NeuronalNet::SignalVector*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr SignalVector_instantiate4(IntPtr other); //const std::vector<float>*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr SignalVector_instantiate5(IntPtr begin, ulong elemCount);
        [DllImport(Net.neuralNetDllName)] static extern void SignalVector_dealocate(IntPtr ptr);

        [DllImport(Net.neuralNetDllName)] static extern IntPtr SignalVector_assign(IntPtr ptr, IntPtr other);
        //[DllImport(Net.neuralNetDllName)] static extern IntPtr SignalVector_element(IntPtr ptr, ulong index); // return float*
        [DllImport(Net.neuralNetDllName)] static extern float SignalVector_getElement(IntPtr ptr, ulong index);
        [DllImport(Net.neuralNetDllName)] static extern void SignalVector_setElement(IntPtr ptr, ulong index, float value);
        [DllImport(Net.neuralNetDllName)] static extern void SignalVector_resize(IntPtr ptr, ulong size);
        [DllImport(Net.neuralNetDllName)] static extern void SignalVector_fill(IntPtr ptr, IntPtr begin, ulong elemCount); //const float* begin

        [DllImport(Net.neuralNetDllName)] static extern ulong SignalVector_size(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern IntPtr SignalVector_begin(IntPtr ptr);// return float*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr SignalVector_end(IntPtr ptr);// return float*

        [DllImport(Net.neuralNetDllName)] static extern void SignalVector_clear(IntPtr ptr);

        [DllImport(Net.neuralNetDllName)] static extern double SignalVector_getSum(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern float SignalVector_getMean(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern float SignalVector_getRootMeanSquare(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern float SignalVector_getGeometricMean(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern float SignalVector_getHarmonicMean(IntPtr ptr);

        public SignalVector()
        {
            vec = SignalVector_instantiate1();
            thisOwner = true;
        }
        public SignalVector(ulong size)
        {
            vec = SignalVector_instantiate2(size);
            thisOwner = true;
        }
        public SignalVector(IntPtr otherVec)
        {
            vec = otherVec; //SignalVector_instantiate3(otherVec);
            thisOwner = false;
        }
        /*public SignalVector(IntPtr otherStdVec)
        {
            vec = SignalVector_instantiate4(otherStdVec);
        }*/
        public SignalVector(IntPtr floatBegin, ulong size)
        {
            vec = SignalVector_instantiate5(floatBegin, size);
            thisOwner = true;
        }
        public SignalVector(SignalVector other)
        {
            vec = SignalVector_instantiate3(other.vec);
            thisOwner = true;
        }
        public SignalVector(List<float> list)
        {
            // pin it to a fixed address:
            
            GCHandle handle = GCHandle.Alloc(list.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                vec = SignalVector_instantiate5(handle.AddrOfPinnedObject(), (ulong)list.Count);
                thisOwner = true;
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }

        ~SignalVector()
        {
            if (thisOwner)
            {
                SignalVector_dealocate(vec);
            }
            vec = IntPtr.Zero;
        }

        public IntPtr GetPtr()
        {
            return vec;
        }
        public SignalVector Assign(SignalVector other)
        {
            SignalVector_assign(vec, other.vec);
            return this;
        }

        public float GetElement(ulong index)
        {
            return SignalVector_getElement(vec, index);
        }
        public void SetElement(ulong index, float value)
        {
            SignalVector_setElement(vec, index, value);
        }
        public void Resize(ulong size)
        {
            SignalVector_resize(vec, size);
        }
        public void Fill(float[] list)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(list, GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                SignalVector_fill(vec, handle.AddrOfPinnedObject(), (ulong)list.Length);
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public void Fill(List<float> list)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(list.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                SignalVector_fill(vec, handle.AddrOfPinnedObject(), (ulong)list.Count);
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public void Fill(float value)
        {
            int size = (int)Size();
            List<float> list = new List<float>(size);
            for (int i = 0; i < size; ++i)
            {
                list.Add(value);
            }
            Fill(list);
        }
        public ulong Size()
        {
            return SignalVector_size(vec);
        }
        public void Clear()
        {
            SignalVector_clear(vec);
        }
        public double GetSum()
        {
            return SignalVector_getSum(vec);
        }
        public float GetMean()
        {
            return SignalVector_getMean(vec);
        }
        public float GetRootMeanSquare()
        {
            return SignalVector_getRootMeanSquare(vec);
        }
        public float GetGemoetricMean()
        {
            return SignalVector_getGeometricMean(vec);
        }
        public float GetHarmonicMean()
        {
            return SignalVector_getHarmonicMean(vec);
        }
    }

}