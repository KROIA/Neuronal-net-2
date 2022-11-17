using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
namespace NeuronalNet
{
    public class NetSerializer
    {
        protected IntPtr serializer = IntPtr.Zero;
        protected bool thisOwner = false;

        [DllImport(Net.neuralNetDllName)] static extern IntPtr NetSerializer_instantiate();
        [DllImport(Net.neuralNetDllName)] static extern void NetSerializer_dealocate(IntPtr serializer);

        [DllImport(Net.neuralNetDllName)] static extern void NetSerializer_setFilePath(IntPtr serializer, IntPtr path); //const char* path
        [DllImport(Net.neuralNetDllName)] static extern IntPtr NetSerializer_getFilePath(IntPtr serializer); // returns const char*

        [DllImport(Net.neuralNetDllName)] static extern bool NetSerializer_saveToFile1(IntPtr serializer, IntPtr net); // NeuronalNet::Net* net
        [DllImport(Net.neuralNetDllName)] static extern bool NetSerializer_saveToFile2(IntPtr serializer, IntPtr net); // NeuronalNet::GeneticNet* net

        [DllImport(Net.neuralNetDllName)] static extern bool NetSerializer_readFromFile1(IntPtr serializer, IntPtr net); // NeuronalNet::Net* net
        [DllImport(Net.neuralNetDllName)] static extern bool NetSerializer_readFromFile2(IntPtr serializer, IntPtr net); // NeuronalNet::GeneticNet* net
    
    
        public NetSerializer()
        {
            serializer = NetSerializer_instantiate();
        }
        public NetSerializer(NetSerializer other)
        {
            serializer = NetSerializer_instantiate();
            SetFilePath(other.GetFilePath());
        }
        ~NetSerializer()
        {
            NetSerializer_dealocate(serializer);
        }

        public void SetFilePath(string path)
        {
            byte[] bytes = System.Text.Encoding.Default.GetBytes(path);
            // pin it to a fixed address:
            GCHandle handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            try
            {
                // retrieve the address as a pointer and use it to call the native method
                NetSerializer_setFilePath(serializer, handle.AddrOfPinnedObject());
            }
            finally
            {
                // free the handle so GC can collect the buffer again
                handle.Free();
            }
            
        }
        public string GetFilePath()
        {
            string str = new string(Marshal.PtrToStringAnsi(NetSerializer_getFilePath(serializer)));
            return str;
        }
        
        public bool SaveToFile(Net net)
        {
            return NetSerializer_saveToFile1(serializer, net.GetPtr());
        }
        public bool SaveToFile(GeneticNet net)
        {
            return NetSerializer_saveToFile2(serializer, net.GetPtr());
        }

        public bool ReadFromFile(Net net)
        {
            return NetSerializer_readFromFile1(serializer, net.GetPtr());
        }
        public bool ReadFromFile(GeneticNet net)
        {
            return NetSerializer_readFromFile2(serializer, net.GetPtr());
        }
    }
}
