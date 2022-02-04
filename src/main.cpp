#include "../inc/net.h"
#include <string>
void printSignal(const SignalVector& sig);

int main(void)
{
	{
		Net net;

		net.setDimensions(3, 3, 20000, 5);
		net.setActivation(Activation::sigmoid);
		net.setHardware(Hardware::cpu);
		net.build();

		net.setInputVector(SignalVector{ 1,1,1 });
		net.calculate();
		net.calculate();
		SignalVector result = net.getOutputVector();
		printSignal(result);


		net.setHardware(Hardware::gpu_cuda);
		net.calculate();
		net.calculate();
		net.calculate();
		net.calculate();
		net.calculate();
		result = net.getOutputVector();
		printSignal(result);
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

