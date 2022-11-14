using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace NeuronalNet
{
    public class GeneticNet
    {
        IntPtr net = IntPtr.Zero;
        bool thisOwner = false;


        [DllImport(Net.neuralNetDllName)] static extern IntPtr GeneticNet_instantiate(ulong netCount);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_dealocate(IntPtr net);


        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setNetCount(IntPtr ptr, ulong netCount);
        [DllImport(Net.neuralNetDllName)] static extern ulong GeneticNet_getNetCount(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setDimensions(IntPtr ptr, ulong inputs, ulong hiddenX, ulong hiddenY, ulong outputs);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setStreamSize(IntPtr ptr, ulong size);
        [DllImport(Net.neuralNetDllName)] static extern ulong GeneticNet_getStreamSize(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern ulong GeneticNet_getInputCount(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern ulong GeneticNet_getHiddenXCount(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern ulong GeneticNet_getHiddenYCount(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern ulong GeneticNet_getOutputCount(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern ulong GeneticNet_getNeuronCount(IntPtr ptr);

        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setActivation(IntPtr ptr, Net.Activation act);
        [DllImport(Net.neuralNetDllName)] static extern Net.Activation GeneticNet_getActivation(IntPtr ptr);

        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setHardware(IntPtr ptr, Net.Hardware ware);
        [DllImport(Net.neuralNetDllName)] static extern Net.Hardware GeneticNet_getHardware(IntPtr ptr);

        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_enableBias(IntPtr ptr, bool enable);
        [DllImport(Net.neuralNetDllName)] static extern bool GeneticNet_isBiasEnabled(IntPtr ptr);

        [DllImport(Net.neuralNetDllName)] static extern bool GeneticNet_build(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern bool GeneticNet_isBuilt(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_randomizeWeights1(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern bool GeneticNet_randomizeWeights2(IntPtr ptr, ulong from, ulong to);
        [DllImport(Net.neuralNetDllName)] static extern float GeneticNet_getRandomValue(float min, float max);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_randomizeBias(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_randomize(IntPtr list, ulong size, float min, float max); // float* list

        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setInputVector1(IntPtr ptr, ulong netIndex, IntPtr signalList); // float* signalList
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setInputVector2(IntPtr ptr, ulong netIndex, ulong stream, IntPtr signalList); // float* signalList
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setInputVector3(IntPtr ptr, ulong netIndex, IntPtr signalList); // const SignalVector* signalList
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setInputVector4(IntPtr ptr, ulong netIndex, ulong stream, IntPtr signalList); // const SignalVector* signalList
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setInputVector5(IntPtr ptr, ulong netIndex, IntPtr streamVector); // const MultiSignalVector* streamVector

        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setInput1(IntPtr ptr, ulong netIndex, ulong input, float signal);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setInput2(IntPtr ptr, ulong netIndex, ulong stream, ulong input, float signal);
        [DllImport(Net.neuralNetDllName)] static extern float GeneticNet_getInput1(IntPtr ptr, ulong netIndex, ulong input);
        [DllImport(Net.neuralNetDllName)] static extern float GeneticNet_getInput2(IntPtr ptr, ulong netIndex, ulong stream, ulong input);
        [DllImport(Net.neuralNetDllName)] static extern IntPtr GeneticNet_getInputVector(IntPtr ptr, ulong netIndex, ulong stream = 0); // returns const SignalVector*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr GeneticNet_getInputStreamVector(IntPtr ptr, ulong netIndex);  // returns const MultiSignalVector*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr GeneticNet_getOutputVector(IntPtr ptr, ulong netIndex, ulong stream = 0); // returns const SignalVector*
        [DllImport(Net.neuralNetDllName)] static extern IntPtr GeneticNet_getOutputStreamVector(IntPtr ptr, ulong netIndex); // returns const MultiSignalVector*
        [DllImport(Net.neuralNetDllName)] static extern float GeneticNet_GetOutput1(IntPtr ptr, ulong netIndex, ulong output);
        [DllImport(Net.neuralNetDllName)] static extern float GeneticNet_GetOutput2(IntPtr ptr, ulong netIndex, ulong stream, ulong output);


        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setWeight1(IntPtr ptr, ulong netIndex, ulong layer, ulong neuron, ulong input, float weight);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setWeight2(IntPtr ptr, ulong netIndex, IntPtr list); //const std::vector<float>* list
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setWeight3(IntPtr ptr, ulong netIndex, IntPtr list); // const float* list
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setWeight4(IntPtr ptr, ulong netIndex, IntPtr list, ulong to); // const float* list
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setWeight5(IntPtr ptr, ulong netIndex, IntPtr list, ulong insertOffset, ulong count); // const float* list
        [DllImport(Net.neuralNetDllName)] static extern float GeneticNet_getWeight1(IntPtr ptr, ulong netIndex, ulong layer, ulong neuron, ulong input);
        [DllImport(Net.neuralNetDllName)] static extern IntPtr GeneticNet_getWeight2(IntPtr ptr, ulong netIndex); // returns const float*
        [DllImport(Net.neuralNetDllName)] static extern ulong GeneticNet_getWeightSize(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setBias1(IntPtr ptr, ulong netIndex, ulong layer, ulong neuron, float bias);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setBias2(IntPtr ptr, ulong netIndex, IntPtr list); // const std::vector<float>* list
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setBias3(IntPtr ptr, ulong netIndex, IntPtr list); // float* list
        [DllImport(Net.neuralNetDllName)] static extern float GeneticNet_getBias1(IntPtr ptr, ulong netIndex, ulong layer, ulong neuron);
        [DllImport(Net.neuralNetDllName)] static extern IntPtr GeneticNet_getBias2(IntPtr ptr, ulong netIndex); // returns const float*



        [DllImport(Net.neuralNetDllName)] static extern IntPtr GeneticNet_getNet(IntPtr ptr, ulong index); // returns NeuronalNet::Net*
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setMutationChance(IntPtr ptr, float chance);
        [DllImport(Net.neuralNetDllName)] static extern float GeneticNet_getMutationChance(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setMutationFactor(IntPtr ptr, float radius);
        [DllImport(Net.neuralNetDllName)] static extern float GeneticNet_getMutationFactor(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_setWeightBounds(IntPtr ptr, float radius);
        [DllImport(Net.neuralNetDllName)] static extern float GeneticNet_getWeightBounds(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_calculate(IntPtr ptr);
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_learn1(IntPtr ptr, IntPtr ranks); //const std::vector<float>* ranks
        [DllImport(Net.neuralNetDllName)] static extern void GeneticNet_learn2(IntPtr ptr, IntPtr ranks);  //const float* ranks



        public GeneticNet(ulong netCount)
        {
            net = GeneticNet_instantiate(netCount);
            thisOwner = true;
        }
        public GeneticNet(IntPtr otherNet)
        {
            net = otherNet;
            thisOwner = false;
        }
        ~GeneticNet()
        {
            if (thisOwner)
                GeneticNet_dealocate(net);
        }

        public IntPtr GetPtr()
        {
            return net;
        }

        public static string GetVersion()
        {
            return Net.GetVersion();
        }
        public static int GetVersion_major()
        {
            return Net.GetVersion_major();
        }
        public static int GetVersion_minor()
        {
            return Net.GetVersion_minor();
        }
        public static int GetVersion_patch()
        {
            return Net.GetVersion_patch();
        }

        public void SetNetCount(ulong netCount)
        {
            GeneticNet_setNetCount(net, netCount);
        }
        public ulong GetNetCount()
        {
            return GeneticNet_getNetCount(net);
        }

        public void SetDimensions(ulong inputs, ulong hiddenX, ulong hiddenY, ulong outputs)
        {
            GeneticNet_setDimensions(net, inputs, hiddenX, hiddenY, outputs);
        }
        public void SetStreamSize(ulong size)
        {
            GeneticNet_setStreamSize(net, size);
        }
        public ulong GetStreamSize()
        {
            return GeneticNet_getStreamSize(net);
        }
        public ulong GetInputCount()
        {
            return GeneticNet_getInputCount(net);
        }
        public ulong GetHiddenXCount()
        {
            return GeneticNet_getHiddenXCount(net);
        }
        public ulong GetHiddenYCount()
        {
            return GeneticNet_getHiddenYCount(net);
        }
        public ulong GetOutputCount()
        {
            return GeneticNet_getOutputCount(net);
        }
        public ulong GetNeuronCount()
        {
            return GeneticNet_getNeuronCount(net);
        }

        public Net.Activation activation
        {
            get
            {
                return GeneticNet_getActivation(net);
            }
            set
            {
                GeneticNet_setActivation(net, value);
            }
        }
        public void SetActivation(Net.Activation act)
        {
            GeneticNet_setActivation(net, act);
        }
        public Net.Activation GetActivation()
        {
            return GeneticNet_getActivation(net);
        }

        public Net.Hardware hardware
        {
            get
            {
                return GeneticNet_getHardware(net);
            }
            set
            {
                GeneticNet_setHardware(net, value);
            }
        }
        public void SetHardware(Net.Hardware ware)
        {
            GeneticNet_setHardware(net, ware);
        }
        public Net.Hardware GetHardware()
        {
            return GeneticNet_getHardware(net);
        }

        public bool enableBias
        {
            get
            {
                return GeneticNet_isBiasEnabled(net);
            }
            set
            {
                GeneticNet_enableBias(net, value);
            }
        }
        public void EnableBias(bool enable)
        {
            GeneticNet_enableBias(net, enable);
        }
        public bool IsBiasEnabled()
        {
            return GeneticNet_isBiasEnabled(net);
        }


        public bool Build()
        {
            return GeneticNet_build(net);
        }
        public bool IsBuilt()
        {
            return GeneticNet_isBuilt(net);
        }

        public void RandomizeWeights()
        {
            GeneticNet_randomizeWeights1(net);
        }
        public bool RandomizeWeights(ulong begin, ulong end)
        {
            return GeneticNet_randomizeWeights2(net, begin, end);
        }
        public static float GetRandomValue(float min, float max)
        {
            return GeneticNet_getRandomValue(min, max);
        }

        public void RandomizeBias()
        {
            GeneticNet_randomizeBias(net);
        }

        public static void Randomize(List<float> list, float min, float max)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(list.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                GeneticNet_randomize(handle.AddrOfPinnedObject(), (ulong)list.Count, min, max);
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public void SetInputVector(ulong netIndex, List<float> signals)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(signals.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                GeneticNet_setInputVector1(net, netIndex, handle.AddrOfPinnedObject());
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public void SetInputVector(ulong netIndex, ulong streamIndex, List<float> signals)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(signals.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                GeneticNet_setInputVector2(net, netIndex, streamIndex, handle.AddrOfPinnedObject());
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public void SetInputVector(ulong netIndex, SignalVector signals)
        {
            GeneticNet_setInputVector3(net, netIndex, signals.GetPtr());
        }
        public void SetInputVector(ulong netIndex, ulong streamIndex, SignalVector signals)
        {
            GeneticNet_setInputVector4(net, netIndex, streamIndex, signals.GetPtr());
        }
        public void SetInputVector(ulong netIndex, MultiSignalVector signals)
        {
            GeneticNet_setInputVector5(net, netIndex, signals.GetPtr());
        }

        public void SetInput(ulong netIndex, ulong input, float signal)
        {
            GeneticNet_setInput1(net, netIndex, input, signal);
        }
        public void SetInput(ulong netIndex, ulong stream, ulong input, float signal)
        {
            GeneticNet_setInput2(net, netIndex, stream, input, signal);
        }
        public float GetInput(ulong netIndex, ulong input)
        {
            return GeneticNet_getInput1(net, netIndex, input);
        }
        public float GetInput(ulong netIndex, ulong stream, ulong input)
        {
            return GeneticNet_getInput2(net, netIndex, stream, input);
        }
        public SignalVector GetInputVector(ulong netIndex, ulong stream = 0)
        {
            IntPtr sig = GeneticNet_getInputVector(net, netIndex, stream);
            if (sig == IntPtr.Zero)
                return null;
            return new SignalVector(sig);
        }
        public MultiSignalVector GetInputStreamVector(ulong netIndex)
        {
            IntPtr sig = GeneticNet_getInputStreamVector(net, netIndex);
            if (sig == IntPtr.Zero)
                return null;
            return new MultiSignalVector(sig);
        }
        public SignalVector GetOutputVector(ulong netIndex, ulong stream = 0)
        {
            IntPtr sig = GeneticNet_getOutputVector(net, netIndex, stream);
            if (sig == IntPtr.Zero)
                return null;
            return new SignalVector(sig);
        }
        public MultiSignalVector GetOutputStreamVector(ulong netIndex)
        {
            IntPtr sig = GeneticNet_getOutputStreamVector(net, netIndex);
            if (sig == IntPtr.Zero)
                return null;
            return new MultiSignalVector(sig);
        }
        public float GetOutput(ulong netIndex, ulong output)
        {
            return GeneticNet_GetOutput1(net, netIndex, output);
        }
        public float GetOutput(ulong netIndex, ulong stream, ulong output)
        {
            return GeneticNet_GetOutput2(net, netIndex, stream, output);
        }

        public void SetWeight(ulong netIndex, ulong layer, ulong neuron, ulong input, float weight)
        {
            GeneticNet_setWeight1(net, netIndex, layer, neuron, input, weight);
        }
        public void SetWeight(ulong netIndex, List<float> weights)
        {
            ulong weightCount = GeneticNet_getWeightSize(net);
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(weights.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                if ((int)weightCount > weights.Count)
                    GeneticNet_setWeight3(net, netIndex, handle.AddrOfPinnedObject());
                else
                    GeneticNet_setWeight4(net, netIndex, handle.AddrOfPinnedObject(), weightCount);
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public void SetWeight(ulong netIndex, List<float> weights, ulong inserPos)
        {
            ulong weightCount = GeneticNet_getWeightSize(net);
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(weights.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                GeneticNet_setWeight5(net, netIndex, handle.AddrOfPinnedObject(), inserPos, (ulong)weights.Count);
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public float GetWeight(ulong netIndex, ulong layer, ulong neuron, ulong input)
        {
            return GeneticNet_getWeight1(net, netIndex, layer, neuron, input);
        }
        public List<float> GetWeight(ulong netIndex)
        {
            ulong weightCount = GeneticNet_getWeightSize(net);
            float[] w = new float[weightCount];
            Marshal.Copy(GeneticNet_getWeight2(net, netIndex), w, 0, (int)weightCount);
            return new List<float>(w);
        }
        public ulong GetWeightSize(ulong netIndex)
        {
            return GeneticNet_getWeightSize(net);
        }
        public void SetBias(ulong netIndex, ulong layer, ulong neuron, float bias)
        {
            GeneticNet_setBias1(net, netIndex, layer, neuron, bias);
        }
        public void SetBias(ulong netIndex, List<float> biasList)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(biasList.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                GeneticNet_setBias3(net, netIndex, handle.AddrOfPinnedObject());
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }
        public float GetBias(ulong netIndex, ulong layer, ulong neuron)
        {
            return GeneticNet_getBias1(net, netIndex, layer, neuron);
        }
        public List<float> GetBias(ulong netIndex)
        {
            ulong neuros = GeneticNet_getNeuronCount(net);
            float[] b = new float[neuros];
            Marshal.Copy(GeneticNet_getBias2(net, netIndex), b, 0, (int)neuros);
            return new List<float>(b);
        }




        public Net GetNet(ulong index)
        {
            IntPtr genNet = GeneticNet_getNet(net, index);
            if (genNet == IntPtr.Zero)
                return null;
            return new Net(genNet);
        }
        public float mutationChance
        {
            get
            {
                return GeneticNet_getMutationChance(net);
            }
            set
            {
                GeneticNet_setMutationChance(net, value);
            }
        }
        public void SetMutationChance(float chance)
        {
            GeneticNet_setMutationChance(net, chance);
        }
        public float GetMutationChance()
        {
            return GeneticNet_getMutationChance(net);
        }

        public float mutationFactor
        {
            get
            {
                return GeneticNet_getMutationFactor(net);
            }
            set
            {
                GeneticNet_setMutationFactor(net, value);
            }
        }
        public void SetMutationFactor(float factor)
        {
            GeneticNet_setMutationFactor(net, factor);
        }
        public float GetMutationFactor()
        {
            return GeneticNet_getMutationFactor(net);
        }
        public float weightBounds
        {
            get
            {
                return GeneticNet_getWeightBounds(net);
            }
            set
            {
                GeneticNet_setWeightBounds(net, value);
            }
        }
        public void SetWeightBounds(float radius)
        {
            GeneticNet_setWeightBounds(net, radius);
        }
        public float GetWeightBounds()
        {
            return GeneticNet_getWeightBounds(net);
        }

        public void Calculate()
        {
            GeneticNet_calculate(net);
        }
        public void Learn(List<float> ranks)
        {
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(ranks.ToArray(), GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                GeneticNet_learn2(net, handle.AddrOfPinnedObject());
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
        }

    }
}