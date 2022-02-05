#include "../inc/net.h"
#include <string>

void printSignal(const SignalVector& sig);

void printBytes(void* ptr, size_t byteCount)
{
	std::cout << "Value @ " << (size_t)ptr;
	for (size_t i = 0; i < byteCount; ++i)
	{
		std::cout << " " << *((short*)ptr);
		ptr = (void*)((size_t)ptr+1);
	}
	std::cout << "\n";
}

void memcopyTest()
{
	struct Storage
	{
		unsigned short a;
		unsigned short b;
	};
	std::cout << "size of short: " << sizeof(short) <<"\n";
	std::cout << "size of Storage: " << sizeof(Storage) << "\n";

	unsigned short test[2];
	test[0] = 0xffff;
	test[1] = 0xffff;

	Storage *stor = (Storage*)&test;
	printBytes((void*)&stor, sizeof(Storage));

	Storage stor2;
	stor2 = *stor;
	printBytes((void*)&stor, sizeof(Storage));

}

int main(void)
{
	nvtxNameOsThread(1,"Main Thread");
	nvtxRangePush(__FUNCTION__);
	{
		//NeuronalNet::GPU_CUDA_getSpecs();
		//memcopyTest();
		//for(int i=0; i<10;i++)
		//	NeuronalNet::GPU_CUDA_memcpyTest();
		//return 0;
		Net net;
		//cudaDeviceProp h_deviceProp;
		//cudaGetDeviceProperties(&h_deviceProp, 0);

		net.setDimensions(3, 100, 4000, 5);
		net.setActivation(Activation::sigmoid);
		net.setHardware(Hardware::cpu);
		net.build();

		net.setInputVector(SignalVector{ 1,1,1 });
		//net.calculate();
	//	net.calculate();
		SignalVector result = net.getOutputVector();
		printSignal(result);


		net.setHardware(Hardware::gpu_cuda);
		//net.calculate();
		//net.calculate();
		//net.calculate();
		net.calculate();
		net.calculate();
		result = net.getOutputVector();
		printSignal(result);
	}

	std::cout << "exiting";
	nvtxRangePop();
	return 0;
}


void printSignal(const SignalVector& sig)
{
	std::cout << "Signals: ";
	for (size_t i = 0; i < sig.size(); ++i)
	{
		printf("%f\t", sig[i]);
	}
	std::cout << "\n";
}

