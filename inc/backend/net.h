#pragma once

#include <iostream>
#include <stdio.h>
#include <vector>

#include "activation.h"
#include "net_kernel.cuh"
#include "multiSignalVector.h"
#include "debug.h"
#include "GraphicsNeuronInterface.h"
#include "GraphicsConnectionInterface.h"
#include "neuronIndex.h"







namespace NeuronalNet
{
	using std::vector;
	enum class Hardware
	{
		cpu,
		gpu_cuda
	};


	class NET_API Net
	{
		public:


		Net();
		~Net();

		void setDimensions(size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs);
		void setStreamSize(size_t size);
		size_t getStreamSize() const;
		size_t getInputCount() const;
		size_t getHiddenXCount() const;
		size_t getHiddenYCount() const;
		size_t getOutputCount() const;

		void setActivation(Activation act);
		Activation getActivation() const;

		void setHardware(enum Hardware ware);
		Hardware getHardware() const;

		virtual bool build();
		bool isBuilt() const;
		void randomizeWeights();
		bool randomizeWeights(size_t from, size_t to);
		static float getRandomValue(float min, float max);
		void randomizeBias();
		void randomize(float* list, size_t size, float min, float max);

		void setInputVector(float* signalList);
		void setInputVector(size_t stream, float* signalList);
		void setInputVector(const SignalVector& signalList);
		void setInputVector(size_t stream, const SignalVector& signalList);
		void setInputVector(const MultiSignalVector& streamVector);

		void setInput(size_t input, float signal);
		void setInput(size_t stream, size_t input, float signal);
		float getInput(size_t input) const;
		float getInput(size_t stream, size_t input) const;
		const SignalVector& getInputVector(size_t stream = 0);
		const MultiSignalVector& getInputStreamVector();
		const SignalVector& getOutputVector(size_t stream = 0);
		const MultiSignalVector& getOutputStreamVector();

		MultiSignalVector getNetinputStreamVector() const;
		MultiSignalVector getNeuronValueStreamVector() const;


		void setWeight(size_t layer, size_t neuron, size_t input, float weight);
		void setWeight(const std::vector<float>& list);
		void setWeight(const float* list);
		void setWeight(const float* list, size_t to);
		void setWeight(const float* list, size_t insertOffset, size_t count);
		float getWeight(size_t layer, size_t neuron, size_t input) const;
		const float* getWeight() const;
		size_t getWeightSize() const;


		void calculate();
		void calculate(size_t stream);
		void calculate(size_t streamBegin, size_t streamEnd);
		void graphics_update(const vector<GraphicsNeuronInterface*> &graphicsNeuronInterfaceList,
							 const vector<GraphicsConnectionInterface*> &graphicsConnectionInterfaceList,
							 size_t streamIndex);

		//void addGraphics(GraphicsNeuronInterface* obj);
		//void removeGraphics(GraphicsNeuronInterface* obj);
		//void addGraphics(GraphicsConnectionInterface* obj);
		//void removeGraphics(GraphicsConnectionInterface* obj);
		//void clearGraphics();

		protected:
		typedef float ActFp(float);

		
		void graphics_update(GraphicsNeuronInterface*obj, size_t streamIndex);
		void graphics_outOfRange(GraphicsNeuronInterface* obj);
		void graphics_update(GraphicsConnectionInterface*obj, size_t streamIndex);
		void graphics_outOfRange(GraphicsConnectionInterface* obj);


		void CPU_calculate(size_t streamBegin, size_t streamEnd); // including begin, excluding end
		void GPU_CUDA_calculate(size_t streamBegin, size_t streamEnd);
		static void CPU_calculateNet(float* weights, float* biasList, float* signals, float* outpuSignals,
									 float* netinputList, float* neuronSignalList,
									 size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, ActFp* activation);
		static void CPU_calculateLayer(float* weights, float* biasList, float* inputSignals,
									   float* netinputList, float* neuronSignalList,
									   size_t neuronCount, size_t inputSignalCount, ActFp* activation);

		void transferWeightsToDevice();
		void transferWeightsToHost();
		void transferSignalsToDevice();
		void transferSignalsToHost();
		void transferBiasToDevice();
		void transferBiasToHost();

		void buildDevice();
		void destroyDevice();
		void buildHostWeights();
		void buildHostBias();
		void destroyHostWeights();
		void destroyHostBias();

		size_t m_inputs;
		size_t m_hiddenX;
		size_t m_hiddenY;
		size_t m_outputs;

		size_t m_streamSize;

		size_t m_neuronCount;
		size_t m_weightsCount;

		Activation m_activation;
		ActFp* m_activationFunc;
		ActFp* m_activationDerivetiveFunc;

		MultiSignalVector m_inputStream;
		MultiSignalVector m_outputStream;
		MultiSignalVector m_netinputList;
		MultiSignalVector m_neuronValueList;

		//float** m_inputSignalList;
		float* m_weightsList;
		float* m_biasList;
		//float** m_outputSingalList;
		bool   m_built;

		// Extern hardware
		Hardware m_hardware;
		float** d_inputSignalList;
		float** h_d_inputSignalList;
		float** d_netinputList;
		float** h_d_netinputList;
		float** d_neuronValueList;
		float** h_d_neuronValueList;
		float* d_weightsList;
		float* d_biasList;
		float** d_outputSingalList;
		float** h_d_outputStream;

		private:
		//vector<GraphicsNeuronInterface*> m_graphicsNeuronInterfaceList;
		//vector<GraphicsConnectionInterface*> m_graphicsConnectionInterfaceList;
		//bool m_useGraphics;

		static float activation_linear(float inp);
		static float activation_finiteLinear(float inp);
		static float activation_binary(float inp);
		static float activation_gauss(float inp);
		static float activation_sigmoid(float inp);

		static float activation_linear_derivetive(float inp);
		static float activation_finiteLinear_derivetive(float inp);
		static float activation_gauss_derivetive(float inp);
		static float activation_sigmoid_derivetive(float inp);

		const static SignalVector m_emptySignalVectorDummy;
		const static MultiSignalVector m_emptyMultiSignalVectorDummy;
	};

};