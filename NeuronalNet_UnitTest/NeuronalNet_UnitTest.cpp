#include "pch.h"
#include "CppUnitTest.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuronalNetUnitTest
{
	using namespace NeuronalNet;
	TEST_CLASS(NeuronalNetUnitTest)
	{
	public:
		
		TEST_METHOD(TestLargeNet)
		{
			POST_CONSOLE_PLOT
			Net net;
			net.setDimensions(3, 100, 100, 5);
			net.setStreamSize(5);
			net.setActivation(Activation::sigmoid);
			net.setHardware(Hardware::cpu);
			net.build();

			net.setInputVector(NeuronalNet::MultiSignalVector({ { 0,0,0 },{ 0,0,1 },{ 0,1,0 },{ 0,1,1 },{ 1,0,0 } }));
			//net.setInputVector(NeuronalNet::MultiSignalVector{ { 1,1,1 },{ 1,1,1 },{ 1,1,1 },{ 1,1,1 },{ 1,1,1 } });
			
			vector<float> cpuWeightsVec1;
			vector<float> cpuBiasVec1;
			getNetData(net, cpuWeightsVec1, cpuBiasVec1);

		/*	const float* w = net.getWeight();
			cpuWeightsVec1.clear();
			cpuWeightsVec1.reserve(net.getWeightSize());
			for (size_t i = 0; i < net.getWeightSize(); ++i)
				cpuWeightsVec1.push_back(w[i]);

			const float* b = net.getBias();
			cpuBiasVec1.clear();
			cpuBiasVec1.reserve(net.getNeuronCount());
			for (size_t i = 0; i < net.getNeuronCount(); ++i)
				cpuBiasVec1.push_back(b[i]);*/

			net.calculate();
			NeuronalNet::MultiSignalVector resultA = net.getOutputStreamVector();
			printSignal(resultA);

			net.setHardware(Hardware::gpu_cuda);
			vector<float> gpuWeightsVec1;
			vector<float> gpuBiasVec1;
			getNetData(net, gpuWeightsVec1, gpuBiasVec1);
			net.calculate();
			NeuronalNet::MultiSignalVector resultB = net.getOutputStreamVector();
			printSignal(resultB);

			net.setHardware(Hardware::cpu);
			vector<float> cpuWeightsVec2;
			vector<float> cpuBiasVec2;
			getNetData(net, cpuWeightsVec2, cpuBiasVec2);
			net.calculate();
			NeuronalNet::MultiSignalVector resultC = net.getOutputStreamVector();
			printSignal(resultC);

			net.setHardware(Hardware::gpu_cuda);
			vector<float> gpuWeightsVec2;
			vector<float> gpuBiasVec2;
			getNetData(net, gpuWeightsVec2, gpuBiasVec2);
			net.calculate();
			NeuronalNet::MultiSignalVector resultD = net.getOutputStreamVector();
			printSignal(resultD);


			// Compare weights
			//Logger::WriteMessage(std::to_string(cpuWeightsVec1.size()).c_str());
			for (size_t i = 0; i < cpuWeightsVec1.size(); ++i)
			{
				if(cpuWeightsVec1[i] != gpuWeightsVec1[i])
					Logger::WriteMessage(("cpuW1["+std::to_string(i)+"] != gpuW1[" + std::to_string(i) + "]: "+
										  std::to_string(cpuWeightsVec1[i])+" != "+ std::to_string(gpuWeightsVec1[i])).c_str());
			}
			for (size_t i = 0; i < cpuWeightsVec2.size(); ++i)
			{
				if (cpuWeightsVec2[i] != gpuWeightsVec2[i])
					Logger::WriteMessage(("cpuW2[" + std::to_string(i) + "] != gpuW2[" + std::to_string(i) + "]: " +
										  std::to_string(cpuWeightsVec2[i]) + " != " + std::to_string(gpuWeightsVec2[i])).c_str());

			}
			// Compare bias
			for (size_t i = 0; i < cpuBiasVec1.size(); ++i)
			{
				if (cpuBiasVec1[i] != gpuBiasVec1[i])
					Logger::WriteMessage(("cpuB1[" + std::to_string(i) + "] != gpuB1[" + std::to_string(i) + "]: " +
										  std::to_string(cpuBiasVec1[i]) + " != " + std::to_string(gpuBiasVec1[i])).c_str());
			}
			for (size_t i = 0; i < cpuBiasVec2.size(); ++i)
			{
				if (cpuBiasVec2[i] != gpuBiasVec2[i])
					Logger::WriteMessage(("cpuB2[" + std::to_string(i) + "] != gpuB2[" + std::to_string(i) + "]: " +
										  std::to_string(cpuBiasVec2[i]) + " != " + std::to_string(gpuBiasVec2[i])).c_str());

			}

			Logger::WriteMessage("Result:\n");
			Assert::IsTrue(signalEqual(resultA, resultB) && 
						   signalEqual(resultA, resultC) &&
						   signalEqual(resultA, resultD),
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
			net.enableBias(false);
			net.build();
			
			size_t weightCount = net.getWeightSize();

			net.setInputVector(NeuronalNet::SignalVector({ 1,1,1 }));

			{
				NeuronalNet::SignalVector checkOutput(std::vector<float>(net.getOutputCount(), 0));
				std::vector<float> weights(weightCount, 0);

				net.setWeight(weights);
				net.calculate();
				NeuronalNet::SignalVector resultA = net.getOutputVector();
				printSignal(resultA); // Shuld be 0
				Logger::WriteMessage("Desired output values: 0\n");
				Assert::IsTrue(signalEqual(resultA, checkOutput),
							   L"T1 Outputs of the calculations are not equal");
			}
			{
				std::vector<float> weights(weightCount, 1);
				// output = 3 * 5 = 15
				NeuronalNet::SignalVector checkOutput(std::vector<float>(net.getOutputCount(), 15));

				net.setWeight(weights);
				net.calculate();
				NeuronalNet::SignalVector resultA = net.getOutputVector();
				printSignal(resultA); // Shuld be 15
				Logger::WriteMessage("Desired output values: 15\n");
				Assert::IsTrue(signalEqual(resultA, checkOutput),
							   L"T2 Outputs of the calculations are not equal");
			}
			

		}
	};
}
