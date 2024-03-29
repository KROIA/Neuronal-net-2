#pragma once

#include <iostream>
#include <stdio.h>
#include <vector>
#include <cstddef>

#include "backend/activation.h"

#include "backend/multiSignalVector.h"
#include "backend/debug.h"
#include "backend/GraphicsNeuronInterface.h"
#include "backend/GraphicsConnectionInterface.h"
#include "backend/neuronIndex.h"
#include "backend/utilities.h"







namespace NeuronalNet
{
	using std::vector;
	enum class Hardware
	{
		cpu,
		gpu_cuda,
		count = 2
	};


	class NET_API Net
	{
		public:


		Net();
        virtual ~Net();

		static std::string getVersion();
		static size_t getVersion_major();
		static size_t getVersion_minor();
		static size_t getVersion_patch();


		virtual void setDimensions(size_t inputs, size_t hiddenX, size_t hiddenY, size_t outputs);
		virtual void setStreamSize(size_t size);
		size_t getStreamSize() const;
		size_t getInputCount() const;
		size_t getHiddenXCount() const;
		size_t getHiddenYCount() const;
		size_t getOutputCount() const;
		size_t getNeuronCount() const;

		virtual void setActivation(Activation act);
		Activation getActivation() const;

		virtual void setHardware(enum Hardware ware);
		Hardware getHardware() const;

		virtual void enableBias(bool enable);
		bool isBiasEnabled() const;

        void unbuild();
		virtual bool build();
		bool isBuilt() const;
        const std::string &getDimensionConfigString() const;
		void randomizeWeights();
		virtual bool randomizeWeights(size_t from, size_t to);
		static float getRandomValue(float min, float max);
        static float map(float x, float inMin,float inMax, float outMin, float outMax);
		void randomizeBias();
		static void randomize(float* list, size_t size, float min, float max);

		virtual void setInputVector(float* signalList);
		virtual void setInputVector(size_t stream, float* signalList);
		virtual void setInputVector(const SignalVector& signalList);
		virtual void setInputVector(size_t stream, const SignalVector& signalList);
		virtual void setInputVector(const MultiSignalVector& streamVector);

		virtual void setInput(size_t input, float signal);
		virtual void setInput(size_t stream, size_t input, float signal);
		virtual float getInput(size_t input) const;
		virtual float getInput(size_t stream, size_t input) const;
		const SignalVector& getInputVector(size_t stream = 0);
		const MultiSignalVector& getInputStreamVector();
		const SignalVector& getOutputVector(size_t stream = 0);
		const MultiSignalVector& getOutputStreamVector();
		float getOutput(size_t output);
		float getOutput(size_t stream, size_t output);

		MultiSignalVector getNetinputStreamVector() const;
		MultiSignalVector getNeuronValueStreamVector() const;


		virtual void setWeight(size_t layer, size_t neuron, size_t input, float weight);
		virtual void setWeight(const std::vector<float>& list);
		virtual void setWeight(const float* list);
		virtual void setWeight(const float* list, size_t to);
		virtual void setWeight(const float* list, size_t insertOffset, size_t count);
		float getWeight(size_t layer, size_t neuron, size_t input) const;
		const float* getWeight() const;
		size_t getWeightSize() const;
		virtual void setBias(size_t layer, size_t neuron, float bias);
		virtual void setBias(const std::vector<float>& list);
		virtual void setBias(const float* list);
		virtual float getBias(size_t layer, size_t neuron);
		const float* getBias() const;


		void calculate();
		void calculate(size_t stream);
		void calculate(size_t streamBegin, size_t streamEnd);
		void graphics_update(const vector<GraphicsNeuronInterface*> &graphicsNeuronInterfaceList,
							 const vector<GraphicsConnectionInterface*> &graphicsConnectionInterfaceList,
							 size_t streamIndex) const;

		//void addGraphics(GraphicsNeuronInterface* obj);
		//void removeGraphics(GraphicsNeuronInterface* obj);
		//void addGraphics(GraphicsConnectionInterface* obj);
		//void removeGraphics(GraphicsConnectionInterface* obj);
		//void clearGraphics();

		protected:
		typedef float ActFp(float);



		
		void graphics_update_CPU(const vector<GraphicsNeuronInterface*>& objList, size_t streamIndex) const;
		void graphics_update_GPU_CUDA(const vector<GraphicsNeuronInterface*>& objList, size_t streamIndex) const;
		inline void graphics_update(GraphicsNeuronInterface *obj,
							 float minN, float maxN, float minO, float maxO,
							 float *inputSignals, float *neuronOutputData, float *netinputData) const;


		inline void graphics_outOfRange(GraphicsNeuronInterface* obj) const;
		void graphics_update_CPU(const vector<GraphicsConnectionInterface*>& objList, size_t streamIndex) const;
		void graphics_update_GPU_CUDA(const vector<GraphicsConnectionInterface*> &objList, size_t streamIndex) const;
		inline void graphics_update(GraphicsConnectionInterface* obj,
							 float minW, float maxW, float minS, float maxS,
							 float* weightList, float* connectionSignalList) const;
		inline void graphics_outOfRange(GraphicsConnectionInterface* obj) const;


		void CPU_calculate(size_t streamBegin, size_t streamEnd); // including begin, excluding end
		void GPU_CUDA_calculate(size_t streamBegin, size_t streamEnd);
        static void CPU_calculateNet(float* weights, float* biasList, float* signalList, float* outpuSignals,
									 float* connectionSignals, float* netinputList, float* neuronSignalList,
									 size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, ActFp* activation);
		static void CPU_calculateLayer(float* weights, float* biasList, float* inputSignals,
									   float* connectionSignals, float* netinputList, float* neuronSignalList,
									   size_t neuronCount, size_t inputSignalCount, ActFp* activation);

		void transferWeightsToDevice();
		void transferWeightsToHost() const;
		void transferSignalsToDevice();
		void transferSignalsToHost();
		void transferBiasToDevice();
		void transferBiasToHost() const;
		void transferNetinputToDevice();
		void transferNetinputToHost();
		void transferNeuronValuesToDevice();
		void transferNeuronValuesToHost();
		void transferWeightSignalProductToDevice();
		void transferWeightSignalProductToHost();


		virtual void buildDevice();
		virtual void destroyDevice();
		void buildHostWeights();
		void buildHostBias();
		void destroyHostWeights();
		void destroyHostBias();

		size_t m_inputs;
		size_t m_hiddenX;
		size_t m_hiddenY;
		size_t m_outputs;
		bool m_useBias;

		size_t m_streamSize;

		size_t m_neuronCount;
		size_t m_weightsCount;

		Activation m_activation;
		ActFp* m_activationFunc;
		ActFp* m_activationDerivetiveFunc;

		MultiSignalVector m_inputStream;
		vector<bool>      m_inputVectorChanged;
		MultiSignalVector m_outputStream;
		MultiSignalVector m_netinputList;
		MultiSignalVector m_weightSignalProduct;
		MultiSignalVector m_neuronValueList;

		//float** m_inputSignalList;
		float* m_weightsList;
		//float** m_weightSignalProduct;
		float* m_biasList;
		//float** m_outputSingalList;
		bool   m_built;
        std::string m_dimensionConfigStr;

		// Extern hardware
		Hardware m_hardware;
		float** d_inputSignalList;
		float** h_d_inputSignalList;
		float** d_netinputList;
		float** h_d_netinputList;
		float** d_neuronValueList;
		float** h_d_neuronValueList;
		mutable float* d_weightsList;
		float** d_weightSignalProduct; // weight * signal for each connection of each signalStream
		float** h_d_weightSignalProduct;
		float* d_biasList;
		float** d_outputSingalList;
		float** h_d_outputStream;


		mutable bool m_weightsChangedFromDeviceTraining;
		mutable bool m_biasChangedFromDeviceTraining;
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

		static size_t m_net_instanceCount;
	};

};
