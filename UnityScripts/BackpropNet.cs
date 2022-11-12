using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace NeuronalNet
{
    public class BackpropNet : Net
    {
        //IntPtr net = IntPtr.Zero;
        //bool thisOwner = false;

        [DllImport(Net.neuralNetDllName)] static extern IntPtr BackpropNet_instantiate();
        [DllImport(Net.neuralNetDllName)] static extern void BackpropNet_dealocate(IntPtr ptr);

        [DllImport(Net.neuralNetDllName)] static extern bool BackpropNet_build(IntPtr ptr);

        [DllImport(Net.neuralNetDllName)] static extern void BackpropNet_setLearnParameter(IntPtr ptr, float learnParam);
        [DllImport(Net.neuralNetDllName)] static extern float BackpropNet_getLearnParameter(IntPtr ptr);

        [DllImport(Net.neuralNetDllName)] static extern void BackpropNet_setExpectedOutput1(IntPtr ptr, IntPtr expectedOutputVec); // const MultiSignalVector* expectedOutputVec
        [DllImport(Net.neuralNetDllName)] static extern void BackpropNet_setExpectedOutput2(IntPtr ptr, IntPtr expectedOutputVec); // const SignalVector* expectedOutputVec
        [DllImport(Net.neuralNetDllName)] static extern void BackpropNet_setExpectedOutput3(IntPtr ptr, ulong streamIndex, IntPtr expectedOutputVec); // const SignalVector* expectedOutputVec
        [DllImport(Net.neuralNetDllName)] static extern void BackpropNet_learn1(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern void BackpropNet_learn2(IntPtr ptr, ulong streamIndex);
        [DllImport(Net.neuralNetDllName)] static extern void BackpropNet_learn3(IntPtr ptr, IntPtr expectedOutputVec); // const MultiSignalVector* expectedOutputVec
        [DllImport(Net.neuralNetDllName)] static extern void BackpropNet_learn4(IntPtr ptr, IntPtr expectedOutputVec); // const SignalVector* expectedOutputVec
        [DllImport(Net.neuralNetDllName)] static extern void BackpropNet_learn5(IntPtr ptr, ulong streamIndex, IntPtr expectedOutputVec); // const SignalVector* expectedOutputVec

        [DllImport(Net.neuralNetDllName)] static extern IntPtr BackpropNet_getError1(IntPtr ptr, ulong streamIndex); // returns SignalVector*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr BackpropNet_getError2(IntPtr ptr, IntPtr expectedOutputVec); // returns const MultiSignalVector*,  const MultiSignalVector* expectedOutputVec
        [DllImport(Net.neuralNetDllName)] static extern IntPtr BackpropNet_getError3(IntPtr ptr, ulong streamIndex, IntPtr expectedOutputVec); // returns SignalVector*,  const SignalVector* expectedOutputVec
        [DllImport(Net.neuralNetDllName)] static extern IntPtr BackpropNet_getError4(IntPtr ptr); // returns const MultiSignalVector*

        public BackpropNet()
        {
            net = BackpropNet_instantiate();
            thisOwner = true;
        }
        public BackpropNet(IntPtr otherNet)
        {
            net = otherNet;
            thisOwner = false;
        }
        ~BackpropNet()
        {
            if (thisOwner)
                BackpropNet_dealocate(net);
        }

        /*public void Build()
        {
            BackpropNet_build(net);
        }*/

        public float learnParameter
        {
            get
            {
                return BackpropNet_getLearnParameter(net);
            }
            set
            {
                BackpropNet_setLearnParameter(net, value);
            }
        }
        public void SetLearnParameter(float value)
        {
            BackpropNet_setLearnParameter(net, value);
        }
        public float GetLearnParameter()
        {
            return BackpropNet_getLearnParameter(net);
        }
        public void SetExpectedOutput(MultiSignalVector expectedOutputStream)
        {
            BackpropNet_setExpectedOutput1(net, expectedOutputStream.GetPtr());
        }
        public void SetExpectedOutput(SignalVector expectedOutput)
        {
            BackpropNet_setExpectedOutput2(net, expectedOutput.GetPtr());
        }
        public void SetExpectedOutput(ulong stream, SignalVector expectedOutput)
        {
            BackpropNet_setExpectedOutput3(net, stream, expectedOutput.GetPtr());
        }
        public void Learn()
        {
            BackpropNet_learn1(net);
        }
        public void Learn(ulong streamIndex)
        {
            BackpropNet_learn2(net, streamIndex);
        }
        public void Learn(MultiSignalVector expectedOutputStream)
        {
            BackpropNet_learn3(net, expectedOutputStream.GetPtr());
        }
        public void Learn(SignalVector expectedOutput)
        {
            BackpropNet_learn4(net, expectedOutput.GetPtr());
        }
        public void Learn(ulong stream, SignalVector expectedOutput)
        {
            BackpropNet_learn5(net, stream, expectedOutput.GetPtr());
        }

        public SignalVector GetError(ulong streamIndex)
        {
            IntPtr err = BackpropNet_getError1(net, streamIndex);
            if (err == IntPtr.Zero)
                return null;
            return new SignalVector(err);
        }
        public MultiSignalVector GetError(MultiSignalVector expectedOutputStream)
        {
            IntPtr err = BackpropNet_getError2(net, expectedOutputStream.GetPtr());
            if (err == IntPtr.Zero)
                return null;
            return new MultiSignalVector(err);
        }
        public SignalVector GetError(ulong streamIndex, SignalVector expectedOutput)
        {
            IntPtr err = BackpropNet_getError3(net, streamIndex, expectedOutput.GetPtr());
            if (err == IntPtr.Zero)
                return null;
            return new SignalVector(err);
        }
        public MultiSignalVector GetError()
        {
            IntPtr err = BackpropNet_getError4(net);
            if (err == IntPtr.Zero)
                return null;
            return new MultiSignalVector(err);
        }
    }
}