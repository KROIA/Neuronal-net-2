using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace NeuronalNet
{
    public class MultiSignalVector
    {
        IntPtr vec = IntPtr.Zero;
        bool thisOwner = false;

        [DllImport(Net.neuralNetDllName)] static extern IntPtr MultiSignalVector_instantiate1();
        [DllImport(Net.neuralNetDllName)] static extern IntPtr MultiSignalVector_instantiate2(IntPtr otherMultiSignalVec); //const MultiSignalVector*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr MultiSignalVector_instantiate3(IntPtr otherStdVec); //const std::vector<SignalVector>*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr MultiSignalVector_instantiate4(IntPtr otherStdVecVec); //const std::vector<std::vector<float>>*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr MultiSignalVector_instantiate5(ulong vectorCount, ulong signalCount);
        [DllImport(Net.neuralNetDllName)] static extern void MultiSignalVector_dealocate(IntPtr ptr);

        [DllImport(Net.neuralNetDllName)] static extern IntPtr MultiSignalVector_assign(IntPtr ptr, IntPtr otherMultiSignalVec); //const MultiSignalVector*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr MultiSignalVector_getElementPtr(IntPtr ptr, ulong vectorIndex);
        [DllImport(Net.neuralNetDllName)] static extern void MultiSignalVector_setElement(IntPtr ptr, ulong vectorIndex, IntPtr elem);

        [DllImport(Net.neuralNetDllName)] static extern void MultiSignalVector_resize1(IntPtr ptr, ulong vectorCount);
        [DllImport(Net.neuralNetDllName)] static extern void MultiSignalVector_resize2(IntPtr ptr, ulong vectorCount, ulong signalCount);
        [DllImport(Net.neuralNetDllName)] static extern void MultiSignalVector_fill1(IntPtr ptr, IntPtr begin, ulong vecCount); //const SignalVector** begin
        [DllImport(Net.neuralNetDllName)] static extern void MultiSignalVector_fill2(IntPtr ptr, IntPtr begin, ulong vecCount); //const SignalVector* begin
        [DllImport(Net.neuralNetDllName)] static extern void MultiSignalVector_fill3(IntPtr ptr, ulong vectorIndex, IntPtr begin, ulong elemCount); //const float* begin
        [DllImport(Net.neuralNetDllName)] static extern void MultiSignalVector_fill4(IntPtr ptr, ulong vectorIndex, IntPtr fillWith); //const SignalVector*



        [DllImport(Net.neuralNetDllName)] static extern ulong MultiSignalVector_size(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern ulong MultiSignalVector_signalSize(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern IntPtr MultiSignalVector_begin(IntPtr ptr); // returns const SignalVector**
        [DllImport(Net.neuralNetDllName)] static extern IntPtr MultiSignalVector_end(IntPtr ptr);   // returns const SignalVector**

        [DllImport(Net.neuralNetDllName)] static extern IntPtr MultiSignalVector_beginGrid(IntPtr ptr); // returns const float**

        [DllImport(Net.neuralNetDllName)] static extern void MultiSignalVector_clear(IntPtr ptr);


        [DllImport(Net.neuralNetDllName)] static extern double MultiSignalVector_getSum(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern float MultiSignalVector_getMean(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern float MultiSignalVector_getRootMeanSquare(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern float MultiSignalVector_getGeometricMean(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern float MultiSignalVector_getHarmonicMean(IntPtr ptr);


        public MultiSignalVector()
        {
            vec = MultiSignalVector_instantiate1();
            thisOwner = true;
        }
        public MultiSignalVector(IntPtr otherVec)
        {
            vec = otherVec;//MultiSignalVector_instantiate2(otherVec);
            thisOwner = false;
        }
        public MultiSignalVector(ulong vectorCount, ulong signalCount)
        {
            vec = MultiSignalVector_instantiate5(vectorCount, signalCount);
            thisOwner = true;
        }
        public MultiSignalVector(List<List<float> > list)
        {
            ulong vecCount = (ulong)list.Count;
            ulong sigCount = 0;
            if (vecCount != 0)
                sigCount = (ulong)list[0].Count;
            vec = MultiSignalVector_instantiate5(vecCount, sigCount);
            thisOwner = true;
            for(ulong i=0; i<vecCount; ++i)
            {
                int index = (int)i;
                if(list[index].Count != (int)sigCount)
                {
                    if (list[index].Count > (int)sigCount)
                    {
                        list[index].RemoveRange((int)sigCount, list[index].Count - (int)sigCount);
                    }
                    else
                    {
                        int toAdd = (int)sigCount - list[index].Count;
                        for(int j=0; j<toAdd; ++j)
                        {
                            list[index].Add(0);
                        }
                    }
                }
                SetElement(i, new SignalVector(list[(int)i]));
            }
        }
        ~MultiSignalVector()
        {
            if (thisOwner)
            {
                MultiSignalVector_dealocate(vec);
            }
            vec = IntPtr.Zero;
        }

        public IntPtr GetPtr()
        {
            return vec;
        }
        public MultiSignalVector Assign(MultiSignalVector other)
        {
            MultiSignalVector_assign(vec, other.vec);
            return this;
        }
        public SignalVector GetElement(ulong index)
        {
            return new SignalVector(MultiSignalVector_getElementPtr(vec, index));
        }
        public void SetElement(ulong index, SignalVector vector)
        {
            MultiSignalVector_setElement(vec, index, vector.GetPtr());
        }

        public void Resize(ulong vectorCount)
        {
            MultiSignalVector_resize1(vec, vectorCount);
        }
        public void Resize(ulong vectorCount, ulong signalCount)
        {
            MultiSignalVector_resize2(vec, vectorCount, signalCount);
        }
        public void Fill(List<SignalVector> list)
        {
            for (int i = 0; i < list.Count; ++i)
            {
                MultiSignalVector_setElement(vec, (ulong)i, list[i].GetPtr());
            }
        }
        public void Fill(float value)
        {
            int size = (int)Size();
            ulong signalCount = MultiSignalVector_size(vec);
            List<SignalVector> list = new List<SignalVector>(size);
            for (int i = 0; i < size; ++i)
            {
                SignalVector sig = new SignalVector(signalCount);
                sig.Fill(value);
                list.Add(sig);
            }
            Fill(list);
        }
        public ulong Size()
        {
            return MultiSignalVector_size(vec);
        }
        public ulong SignalSize()
        {
            return MultiSignalVector_signalSize(vec);
        }
        public void Clear()
        {
            MultiSignalVector_clear(vec);
        }
        public double GetSum()
        {
            return MultiSignalVector_getSum(vec);
        }
        public float GetMean()
        {
            return MultiSignalVector_getMean(vec);
        }
        public float GetRootMeanSquare()
        {
            return MultiSignalVector_getRootMeanSquare(vec);
        }
        public float GetGemoetricMean()
        {
            return MultiSignalVector_getGeometricMean(vec);
        }
        public float GetHarmonicMean()
        {
            return MultiSignalVector_getHarmonicMean(vec);
        }

    }
}