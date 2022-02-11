#include "../inc/net.h"
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>

using std::vector;
using std::cout;
using std::string;

sf::RenderWindow* window;

void printSignal(const SignalVector& sig);
bool signalEqual(const SignalVector& a, const SignalVector& b);
string getByteString(double bytes);


int main(void)
{
	//window = new sf::RenderWindow(sf::VideoMode(1000, 1000), "My window");
	//NeuronalNet::testCUDA();
	//return 0;
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

		net.setInputVector(SignalVector{ 0,1,1 });
		//net.calculate();
		net.calculate();
		net.calculate();
		net.calculate();
		net.calculate();
		cout << "\n";
		SignalVector resultA = net.getOutputVector();
		printSignal(resultA);


		net.setHardware(Hardware::gpu_cuda);
		//net.calculate();
		//net.calculate();
		//net.calculate();
		net.calculate();
		net.calculate();
		net.calculate();
		net.calculate();
		cout << "\n";
		SignalVector resultB = net.getOutputVector();
		printSignal(resultB);

		net.setHardware(Hardware::cpu);
		net.calculate();
		net.calculate();
		net.calculate();
		cout << "\n";
		SignalVector resultC = net.getOutputVector();
		printSignal(resultC);

		cout << "Result:\n";
		if (signalEqual(resultA, resultB) && signalEqual(resultA, resultC))
			cout << "  PASS\n";
		else
			cout << "  FAIL Output vectors are not equal\n";
	}

	std::cout << "exiting";
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

string getByteString(double bytes)
{
	double b = bytes;
	int exp = 0;

	while (b >= 1000.f)
	{
		b /= 1000;
		exp += 3;
	}
	switch (exp)
	{
		case 0:
			return std::to_string(b) + " bytes";
		case 3:
			return std::to_string(b) + " KB";
		case 6:
			return std::to_string(b) + " MB";
		case 9:
			return std::to_string(b) + " GB";
		case 12:
			return std::to_string(b) + " TB";
	}
	return "unknown amount";
}

bool signalEqual(const SignalVector& a, const SignalVector& b)
{
	if (a.size() != b.size())
		return false;

	for (size_t i = 0; i < a.size(); ++i)
	{
		if (abs(a[i] - b[i]) > 0.00001)
			return false;
	}
	return true;
}