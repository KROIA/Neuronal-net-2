#include "pch.h"
#include "CppUnitTest.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace CudaUnitTest
{
	//using namespace NeuronalNet;
	TEST_CLASS(CudaUnitTest)
	{
		public:

		TEST_METHOD(MemorySwitch)
		{
			/*Net net;
			net.setDimensions(3, 100, 4000, 5);
			net.setStreamSize(5);
			net.setActivation(Activation::sigmoid);
			net.setHardware(Hardware::cpu);
			net.build();*/

			NeuronalNet::SignalVector vec(1);

			float* d_buffer = nullptr;
			size_t elements = 10;
			NeuronalNet::GPU_CUDA_allocMem(d_buffer, elements * sizeof(float));
			CUDA_validateLastError();
			Assert::IsNotNull(d_buffer, L"Allocation failed");

			float* h_buffer = new float[elements];
			for (size_t i = 0; i < elements; ++i)
			{
				h_buffer[i] = rand() % 100;
			}

			NeuronalNet::GPU_CUDA_transferToDevice(d_buffer, h_buffer, elements * sizeof(float));
			CUDA_validateLastError();

			float* h_buffer2 = new float[elements];
			NeuronalNet::GPU_CUDA_transferToHost(d_buffer, h_buffer2, elements * sizeof(float));
			CUDA_validateLastError();


			for (size_t i = 0; i < elements; ++i)
			{
				double dif = abs(h_buffer[i] - h_buffer2[i]);
				Assert::IsFalse(dif > 0.00001, (std::wstring(L"Buffer not copied correct ") + std::to_wstring(h_buffer[i]) + std::wstring(L" != ") + std::to_wstring(h_buffer2[i])).c_str());
			}

			NeuronalNet::GPU_CUDA_freeMem(d_buffer);
			CUDA_validateLastError();
			delete[] h_buffer;
			delete[] h_buffer2;
			plotConsoleOutput();
		}
	};
}