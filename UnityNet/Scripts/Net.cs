using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace NeuronalNet
{
    public class Net
    {
        public enum Hardware
        {
            cpu,
            gpu_cuda
        };

        public enum Activation
        {
            linear,
            finiteLinear,
            binary,
            sigmoid,
            gauss
        };

        protected IntPtr net = IntPtr.Zero;
        protected bool thisOwner = false;
        public const string neuralNetDllName = "Neural-net-2.dll";

        [DllImport(neuralNetDllName)] static extern IntPtr Net_instantiate();
        [DllImport(neuralNetDllName)] static extern void Net_dealocate(IntPtr net);
        [DllImport(neuralNetDllName)] static extern ulong Net_getVersion_major();
        [DllImport(neuralNetDllName)] static extern ulong Net_getVersion_minor();
        [DllImport(neuralNetDllName)] static extern ulong Net_getVersion_patch();
        [DllImport(neuralNetDllName)] static extern void Net_setDimensions(IntPtr ptr, ulong inputs, ulong hiddenX, ulong hiddenY, ulong outputs);
        [DllImport(neuralNetDllName)] static extern void Net_setStreamSize(IntPtr ptr, ulong size);
        [DllImport(neuralNetDllName)] static extern ulong Net_getStreamSize(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern ulong Net_getInputCount(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern ulong Net_getHiddenXCount(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern ulong Net_getHiddenYCount(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern ulong Net_getOutputCount(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern ulong Net_getNeuronCount(IntPtr ptr);

        [DllImport(neuralNetDllName)] static extern void Net_setActivation(IntPtr ptr, Activation act);
        [DllImport(neuralNetDllName)] static extern Activation Net_getActivation(IntPtr ptr);

        [DllImport(neuralNetDllName)] static extern void Net_setHardware(IntPtr ptr, Hardware ware);
        [DllImport(neuralNetDllName)] static extern Hardware Net_getHardware(IntPtr ptr);

        [DllImport(neuralNetDllName)] static extern void Net_enableBias(IntPtr ptr, bool enable);
        [DllImport(neuralNetDllName)] static extern bool Net_isBiasEnabled(IntPtr ptr);

        [DllImport(neuralNetDllName)] static extern bool Net_build(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern bool Net_isBuilt(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern void Net_randomizeWeights1(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern bool Net_randomizeWeights2(IntPtr ptr, ulong from, ulong to);
        [DllImport(neuralNetDllName)] static extern float Net_getRandomValue(float min, float max);
        [DllImport(neuralNetDllName)] static extern void Net_randomizeBias(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern void Net_randomize(IntPtr list, ulong size, float min, float max); // float* list

        [DllImport(neuralNetDllName)] static extern void Net_setInputVector1(IntPtr ptr, IntPtr signalList); // float* signalList
        [DllImport(neuralNetDllName)] static extern void Net_setInputVector2(IntPtr ptr, ulong stream, IntPtr signalList); // float* signalList
        [DllImport(neuralNetDllName)] static extern void Net_setInputVector3(IntPtr ptr, IntPtr signalList); // const SignalVector* signalList
        [DllImport(neuralNetDllName)] static extern void Net_setInputVector4(IntPtr ptr, ulong stream, IntPtr signalList); // const SignalVector* signalList
        [DllImport(neuralNetDllName)] static extern void Net_setInputVector5(IntPtr ptr, IntPtr streamVector); // const MultiSignalVector* streamVector

        [DllImport(neuralNetDllName)] static extern void Net_setInput1(IntPtr ptr, ulong input, float signal);
        [DllImport(neuralNetDllName)] static extern void Net_setInput2(IntPtr ptr, ulong stream, ulong input, float signal);
        [DllImport(neuralNetDllName)] static extern float Net_getInput1(IntPtr ptr, ulong input);
        [DllImport(neuralNetDllName)] static extern float Net_getInput2(IntPtr ptr, ulong stream, ulong input);
        [DllImport(neuralNetDllName)] static extern IntPtr Net_getInputVector(IntPtr ptr, ulong stream = 0); // returns const SignalVector*
        [DllImport(neuralNetDllName)] static extern IntPtr Net_getInputStreamVector(IntPtr ptr);  // returns const MultiSignalVector*
        [DllImport(neuralNetDllName)] static extern IntPtr Net_getOutputVector(IntPtr ptr, ulong stream = 0); // returns const SignalVector*
        [DllImport(neuralNetDllName)] static extern IntPtr Net_getOutputStreamVector(IntPtr ptr); // returns const MultiSignalVector*
        [DllImport(neuralNetDllName)] static extern float Net_GetOutput1(IntPtr ptr, ulong output);
        [DllImport(neuralNetDllName)] static extern float Net_GetOutput2(IntPtr ptr, ulong stream, ulong output);


        [DllImport(neuralNetDllName)] static extern void Net_setWeight1(IntPtr ptr, ulong layer, ulong neuron, ulong input, float weight);
        [DllImport(neuralNetDllName)] static extern void Net_setWeight2(IntPtr ptr, IntPtr list); //const std::vector<float>* list
        [DllImport(neuralNetDllName)] static extern void Net_setWeight3(IntPtr ptr, IntPtr list); // const float* list
        [DllImport(neuralNetDllName)] static extern void Net_setWeight4(IntPtr ptr, IntPtr list, ulong to); // const float* list
        [DllImport(neuralNetDllName)] static extern void Net_setWeight5(IntPtr ptr, IntPtr list, ulong insertOffset, ulong count); // const float* list
        [DllImport(neuralNetDllName)] static extern float Net_getWeight1(IntPtr ptr, ulong layer, ulong neuron, ulong input);
        [DllImport(neuralNetDllName)] static extern IntPtr Net_getWeight2(IntPtr ptr); // returns const float*
        [DllImport(neuralNetDllName)] static extern ulong Net_getWeightSize(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern void Net_setBias1(IntPtr ptr, ulong layer, ulong neuron, float bias);
        [DllImport(neuralNetDllName)] static extern void Net_setBias2(IntPtr ptr, IntPtr list); // const std::vector<float>* list
        [DllImport(neuralNetDllName)] static extern void Net_setBias3(IntPtr ptr, IntPtr list); // float* list
        [DllImport(neuralNetDllName)] static extern float Net_getBias1(IntPtr ptr, ulong layer, ulong neuron);
        [DllImport(neuralNetDllName)] static extern IntPtr Net_getBias2(IntPtr ptr); // returns const float*


        [DllImport(neuralNetDllName)] static extern void Net_calculate1(IntPtr ptr);
        [DllImport(neuralNetDllName)] static extern void Net_calculate2(IntPtr ptr, ulong stream);
        [DllImport(neuralNetDllName)] static extern void Net_calculate3(IntPtr ptr, ulong streamBegin, ulong streamEnd);
        public Net()
        {
            net = Net_instantiate();
            thisOwner = true;
        }
        public Net(IntPtr otherNet)
        {
            net = otherNet;
            thisOwner = false;
        }
        ~Net()
        {
            if (thisOwner)
                Net_dealocate(net);
        }

        public IntPtr GetPtr()
        {
            return net;
        }

        public static string GetVersion()
        {
            return GetVersion_major() + "." +
                   GetVersion_minor() + "." +
                   GetVersion_patch();
        }
        public static int GetVersion_major()
        {
            return (int)Net_getVersion_major();
        }
        public static int GetVersion_minor()
        {
            return (int)Net_getVersion_minor();
        }
        public static int GetVersion_patch()
        {
            return (int)Net_getVersion_patch();
        }


        public void SetDimensions(ulong inputs, ulong hiddenX, ulong hiddenY, ulong outputs)
        {
            Net_setDimensions(net, inputs, hiddenX, hiddenY, outputs);
        }
        public void SetStreamSize(ulong size)
        {
            Net_setStreamSize(net, size);
        }
        public ulong GetStreamSize()
        {
            return Net_getStreamSize(net);
        }
        public ulong GetInputCount()
        {
            return Net_getInputCount(net);
        }
        public ulong GetHiddenXCount()
        {
            return Net_getHiddenXCount(net);
        }
        public ulong GetHiddenYCount()
        {
            return Net_getHiddenYCount(net);
        }
        public ulong GetOutputCount()
        {
            return Net_getOutputCount(net);
        }
        public ulong GetNeuronCount()
        {
            return Net_getNeuronCount(net);
        }

        public Activation activation
        {
            get
            {
                return Net_getActivation(net);
            }
            set
            {
                Net_setActivation(net, value);
            }
        }
        public void SetActivation(Activation act)
        {
            Net_setActivation(net, act);
        }
        public Activation GetActivation()
        {
            return Net_getActivation(net);
        }

        public Hardware hardware
        {
            get
            {
                return Net_getHardware(net);
            }
            set
            {
                Net_setHardware(net, value);
            }
        }
        public void SetHardware(Hardware ware)
        {
            Net_setHardware(net, ware);
        }
        public Hardware GetHardware()
        {
            return Net_getHardware(net);
        }

        public bool enableBias
        {
            get
            {
                return Net_isBiasEnabled(net);
            }
            set
            {
                Net_enableBias(net, value);
            }
        }
        public void EnableBias(bool enable)
        {
            Net_enableBias(net, enable);
        }
        public bool IsBiasEnabled()
        {
            return Net_isBiasEnabled(net);
        }


        public bool Build()
        {
            return Net_build(net);
        }
        public bool IsBuilt()
        {
            return Net_isBuilt(net);
        }

        public void RandomizeWeights()
        {
            Net_randomizeWeights1(net);
        }
        public bool RandomizeWeights(ulong begin, ulong end)
        {
            return Net_randomizeWeights2(net, begin, end);
        }
        public static float GetRandomValue(float min, float max)
        {
            return Net_getRandomValue(min, max);
        }

        public void RandomizeBias()
        {
            Net_randomizeBias(net);
        }

        public static void Randomize(List<float> list, float min, float max)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(list.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                Net_randomize(handle.AddrOfPinnedObject(), (ulong)list.Count, min, max);
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public void SetInputVector(List<float> signals)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(signals.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                Net_setInputVector1(net, handle.AddrOfPinnedObject());
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public void SetInputVector(ulong streamIndex, List<float> signals)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(signals.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                Net_setInputVector2(net, streamIndex, handle.AddrOfPinnedObject());
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public void SetInputVector(SignalVector signals)
        {
            Net_setInputVector3(net, signals.GetPtr());
        }
        public void SetInputVector(ulong streamIndex, SignalVector signals)
        {
            Net_setInputVector4(net, streamIndex, signals.GetPtr());
        }
        public void SetInputVector(MultiSignalVector signals)
        {
            Net_setInputVector5(net, signals.GetPtr());
        }

        public void SetInput(ulong input, float signal)
        {
            Net_setInput1(net, input, signal);
        }
        public void SetInput(ulong stream, ulong input, float signal)
        {
            Net_setInput2(net, stream, input, signal);
        }
        public float GetInput(ulong input)
        {
            return Net_getInput1(net, input);
        }
        public float GetInput(ulong stream, ulong input)
        {
            return Net_getInput2(net, stream, input);
        }
        public SignalVector GetInputVector(ulong stream = 0)
        {
            IntPtr sig = Net_getInputVector(net, stream);
            if (sig == IntPtr.Zero)
                return null;
            return new SignalVector(sig);
        }
        public MultiSignalVector GetInputStreamVector()
        {
            IntPtr sig = Net_getInputStreamVector(net);
            if (sig == IntPtr.Zero)
                return null;
            return new MultiSignalVector(sig);
        }
        public SignalVector GetOutputVector(ulong stream = 0)
        {
            IntPtr sig = Net_getOutputVector(net, stream);
            if (sig == IntPtr.Zero)
                return null;
            return new SignalVector(sig);
        }
        public MultiSignalVector GetOutputStreamVector()
        {
            IntPtr sig = Net_getOutputStreamVector(net);
            if (sig == IntPtr.Zero)
                return null;
            return new MultiSignalVector(sig);
        }
        public float GetOutput(ulong output)
        {
            return Net_GetOutput1(net, output);
        }
        public float GetOutput(ulong stream, ulong output)
        {
            return Net_GetOutput2(net, stream, output);
        }

        public void SetWeight(ulong layer, ulong neuron, ulong input, float weight)
        {
            Net_setWeight1(net, layer, neuron, input, weight);
        }
        public void SetWeight(List<float> weights)
        {
            ulong weightCount = Net_getWeightSize(net);
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(weights.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                if ((int)weightCount > weights.Count)
                    Net_setWeight3(net, handle.AddrOfPinnedObject());
                else
                    Net_setWeight4(net, handle.AddrOfPinnedObject(), weightCount);
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public void SetWeight(List<float> weights, ulong inserPos)
        {
            ulong weightCount = Net_getWeightSize(net);
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(weights.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                Net_setWeight5(net, handle.AddrOfPinnedObject(), inserPos, (ulong)weights.Count);
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public float GetWeight(ulong layer, ulong neuron, ulong input)
        {
            return Net_getWeight1(net, layer, neuron, input);
        }
        public List<float> GetWeight()
        {
            ulong weightCount = Net_getWeightSize(net);
            float[] w = new float[weightCount];
            Marshal.Copy(Net_getWeight2(net), w, 0, (int)weightCount);
            return new List<float>(w);
        }
        public ulong GetWeightSize()
        {
            return Net_getWeightSize(net);
        }
        public void SetBias(ulong layer, ulong neuron, float bias)
        {
            Net_setBias1(net, layer, neuron, bias);
        }
        public void SetBias(List<float> biasList)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(biasList.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                Net_setBias3(net, handle.AddrOfPinnedObject());
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public float GetBias(ulong layer, ulong neuron)
        {
            return Net_getBias1(net, layer, neuron);
        }
        public List<float> GetBias()
        {
            ulong neuros = Net_getNeuronCount(net);
            float[] b = new float[neuros];
            Marshal.Copy(Net_getBias2(net), b, 0, (int)neuros);
            return new List<float>(b);
        }

        public void Calculate()
        {
            Net_calculate1(net);
        }
        public void Calculate(ulong stream)
        {
            Net_calculate2(net, stream);
        }
        public void Calculate(ulong streamBegin, ulong streamEnd)
        {
            Net_calculate3(net, streamBegin, streamEnd);
        }
    }

}