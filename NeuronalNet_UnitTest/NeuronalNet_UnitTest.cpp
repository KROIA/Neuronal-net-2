#include "pch.h"
#include "CppUnitTest.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuronalNetUnitTest
{
	TEST_CLASS(NeuronalNetUnitTest)
	{
	public:
		
		TEST_METHOD(TestLargeNet)
		{
			POST_CONSOLE_PLOT
			Net net;
			net.setDimensions(3, 100, 4000, 5);
			net.setStreamSize(5);
			net.setActivation(Activation::sigmoid);
			net.setHardware(Hardware::cpu);
			net.build();

			net.setInputVector(MultiSignalVector({ { 0,0,0 },{ 0,0,1 },{ 0,1,0 },{ 0,1,1 },{ 1,0,0 } }));
			//net.setInputVector(MultiSignalVector{ { 1,1,1 },{ 1,1,1 },{ 1,1,1 },{ 1,1,1 },{ 1,1,1 } });
			net.calculate();
			MultiSignalVector resultA = net.getOutputStreamVector();
			printSignal(resultA);

			net.setHardware(Hardware::gpu_cuda);
			net.calculate();
			MultiSignalVector resultB = net.getOutputStreamVector();
			printSignal(resultB);

			//net.setHardware(Hardware::cpu);
			//net.calculate();
			MultiSignalVector resultC = net.getOutputStreamVector();
			printSignal(resultC);

			Logger::WriteMessage("Result:\n");
			Assert::IsTrue(signalEqual(resultA, resultB) && signalEqual(resultA, resultC),
							L"Outputs of the calculations are not equal");
			
		}
		TEST_METHOD(Basic_1)
		{
			POST_CONSOLE_PLOT
			Net net;
			net.setDimensions(3, 1, 5, 5);
			net.setStreamSize(1);
			net.setActivation(Activation::linear);
			net.setHardware(Hardware::cpu);
			net.build();
			size_t weightCount = net.getWeightSize();

			net.setInputVector(SignalVector({ 1,1,1 }));

			{
				SignalVector checkOutput(std::vector<float>(net.getOutputCount(), 0));
				std::vector<float> weights(weightCount, 0);

				net.setWeight(weights);
				net.calculate();
				SignalVector resultA = net.getOutputVector();
				printSignal(resultA); // Shuld be 0
				Logger::WriteMessage("Desired output values: 0\n");
				Assert::IsTrue(signalEqual(resultA, checkOutput),
							   L"T1 Outputs of the calculations are not equal");
			}
			{
				std::vector<float> weights(weightCount, 1);
				// output = 3 * 5 = 15
				SignalVector checkOutput(std::vector<float>(net.getOutputCount(), 15));

				net.setWeight(weights);
				net.calculate();
				SignalVector resultA = net.getOutputVector();
				printSignal(resultA); // Shuld be 15
				Logger::WriteMessage("Desired output values: 15\n");
				Assert::IsTrue(signalEqual(resultA, checkOutput),
							   L"T2 Outputs of the calculations are not equal");
			}
			

		}
	};
}
