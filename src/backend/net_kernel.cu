#include "backend/net_kernel.cuh"


#define BLOCK_DIM 16

namespace NeuronalNet 
{
    // Kernel global var.
    CUDA_info* _d_cudaInfo = nullptr;
    CUDA_info* _h_cudaInfo = nullptr;

    __host__ 
        void testCUDA()
    {
        size_t w = 3;
        size_t h = 5;
        float* matrix = DBG_NEW float[w * h];
        /*for (size_t y = 0; y < w * h; ++y)
        {
            matrix[y] = 0;
        }*/
        for (size_t y = 0; y < h; ++y)
        {
            for (size_t x = 0; x < w; ++x)
            {
                matrix[y * w + x] = y * w + x;
            }
        }
        printf("Original Matrix:\n");
        printf("V1:\n");
        for (size_t y = 0; y < h; ++y)
        {
            for (size_t x = 0; x < w; ++x)
            {
                printf("%3.0f ", matrix[y * w + x]);
            }
            printf("\n");
        }
        printf("V2:\n");
        for (size_t y = 0; y < w; ++y)
        {
            for (size_t x = 0; x < h; ++x)
            {
                printf("%3.0f ", matrix[y * h + x]);
            }
            printf("\n");
        }

        float* dMatrix;
        cudaMalloc(&dMatrix, w * h * sizeof(float));
        GPU_CUDA_transferToDevice(dMatrix, matrix, w * h * sizeof(float));
        kernel_transposeMatrix << <1, 1 >> > (dMatrix, w, h, _d_cudaInfo);
        for (size_t y = 0; y < w*h; ++y)
        {
            matrix[y] = 0;
        }
        cuda_handleError(cudaDeviceSynchronize());

        GPU_CUDA_transferToHost(dMatrix, matrix, w * h * sizeof(float));

        printf("Transposed Matrix:\n");
        printf("V1:\n");
        for (size_t y = 0; y < h; ++y)
        {
            for (size_t x = 0; x < w; ++x)
            {
                printf("%3.0f ", matrix[y * w + x]);
            }
            printf("\n");
        }
        printf("V2:\n");
        for (size_t y = 0; y < w; ++y)
        {
            for (size_t x = 0; x < h; ++x)
            {
                printf("%3.0f ",matrix[y * h + x]);
            }
            printf("\n");
        }

        cudaFree(dMatrix);
        delete[] matrix;
    }


    __host__ 
        cudaDeviceProp GPU_CUDA_getSpecs()
    {
        cudaDeviceProp h_deviceProp;
        cudaGetDeviceProperties(&h_deviceProp, 0);
        if (_d_cudaInfo == nullptr)
        {
            cudaMalloc(&_d_cudaInfo, sizeof(CUDA_info));
            CUDA_info info;
            info.maxThreadDim.x = h_deviceProp.maxThreadsDim[0];
            info.maxThreadDim.y = h_deviceProp.maxThreadsDim[1];
            info.maxThreadDim.z = h_deviceProp.maxThreadsDim[2];
            info.maxThreadsPerBlock = h_deviceProp.maxThreadsPerBlock;
            info.totalGlobalMemory = h_deviceProp.totalGlobalMem;

            cudaMemcpy(_d_cudaInfo, &info, sizeof(CUDA_info), cudaMemcpyHostToDevice);
        }
        if (_h_cudaInfo == nullptr)
        {
            _h_cudaInfo = DBG_NEW CUDA_info;
            _h_cudaInfo->maxThreadDim.x = h_deviceProp.maxThreadsDim[0];
            _h_cudaInfo->maxThreadDim.y = h_deviceProp.maxThreadsDim[1];
            _h_cudaInfo->maxThreadDim.z = h_deviceProp.maxThreadsDim[2];
            _h_cudaInfo->maxThreadsPerBlock = h_deviceProp.maxThreadsPerBlock;
            _h_cudaInfo->totalGlobalMemory = h_deviceProp.totalGlobalMem;
        }
        return h_deviceProp;
    }
    __host__
        void GPU_CUDA_deleteSpecs()
    {
        if (_d_cudaInfo)
        {
            cudaFree(_d_cudaInfo);
            _d_cudaInfo = nullptr;
        }
        if (_h_cudaInfo)
        {
            delete _h_cudaInfo;
            _h_cudaInfo = nullptr;
        }
    }
    __host__ 
        void GPU_CUDA_calculateNet(float* weights, float* biasList, float** multiSignalVec, float** multiOutputVec, 
                                   float** multiConnectionSignalList, float** multiNetinputList, float** multiNeuronSignalList, size_t multiSignalSize,
                                   size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation activation,
                                   CUDA_info* d_info)
    {

        kernel_calculateNet << <1, 1 >> > (weights, biasList, multiSignalVec, multiOutputVec, 
                                           multiConnectionSignalList, multiNetinputList, multiNeuronSignalList, multiSignalSize,
                                           inputCount, hiddenX, hiddenY, outputCount, activation,
                                           d_info);
        cuda_handleError(cudaDeviceSynchronize());
    }
    __host__
        void GPU_CUDA_getRandomValues(float* h_list, size_t elements, float min, float max)
    {
        size_t blockSize = _h_cudaInfo->maxThreadsPerBlock;
        size_t numBlocks = (elements - 1) / blockSize + 1;
        curandGenerator_t gen;
        float* d_list;
        cudaMalloc(&d_list, elements * sizeof(float));
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_list, elements);
        cuda_handleError(cudaDeviceSynchronize());
        kernel_offsetScale << <numBlocks, blockSize >> > (d_list, -0.5, 2, elements, _d_cudaInfo);
        cuda_handleError(cudaDeviceSynchronize());
        cudaMemcpy(h_list, d_list, elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_list);
    }

    
    template <typename T>
    __host__ void GPU_CUDA_allocMem(T*& d_list, size_t byteCount)
    {
        d_list = nullptr;
        cuda_handleError(cudaMalloc(&d_list, byteCount));
    }
    template NET_API __host__ void GPU_CUDA_allocMem<float>(float*& d_list, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_allocMem<float*>(float**& d_list, size_t byteCount);

    template <typename T>
    __host__ void GPU_CUDA_freeMem(T*& d_list)
    {
        if (!d_list)
            return;
        cudaError_t err = cudaFree(d_list);
        cuda_handleError(err);
        if (err == cudaError::cudaSuccess)
            d_list = nullptr;
    }
    template NET_API __host__ void GPU_CUDA_freeMem<float>(float*& d_list);
    template NET_API __host__ void GPU_CUDA_freeMem<float*>(float**& d_list);

    template <typename T>
    __host__ void GPU_CUDA_memset(T*& d_list, int value, size_t byteCount)
    {
        cudaMemset(d_list, value, byteCount);
    }
    template NET_API __host__ void GPU_CUDA_memset<float>(float*& d_list, int value, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_memset<float*>(float**& d_list, int value, size_t byteCount);

    template <typename T>
    __host__ void GPU_CUDA_memcpy(T*& d_source, T*& d_destination, size_t byteCount)
    {
        cudaMemcpy(d_destination, d_source, byteCount, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }
    template NET_API __host__ void GPU_CUDA_memcpy<float>(float*& d_source, float*& d_destination, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_memcpy<float*>(float**& d_source, float**& d_destination, size_t byteCount);


    template <typename T>
    __host__ void GPU_CUDA_transferToDevice(T* d_list, T* h_list, size_t byteCount)
    {
        cuda_handleError(cudaMemcpy((void*)d_list, (void*)h_list, byteCount, cudaMemcpyHostToDevice));
    }
    template NET_API __host__ void GPU_CUDA_transferToDevice<float>(float* d_list, float* h_list, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_transferToDevice<float*>(float** d_list, float** h_list, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_transferToDevice<const float>(const float* d_list, const float* h_list, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_transferToDevice<const float*>(const float** d_list, const float** h_list, size_t byteCount);


    template <typename T>
    __host__ void GPU_CUDA_transferToHost(T* d_list, T* h_list, size_t byteCount)
    {
        cuda_handleError(cudaMemcpy((void *)h_list, (void*)d_list, byteCount, cudaMemcpyDeviceToHost));
    }
    template NET_API __host__ void GPU_CUDA_transferToHost<float>(float* d_list, float* h_list, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_transferToHost<float*>(float** d_list, float** h_list, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_transferToHost<const float>(const float* d_list, const float* h_list, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_transferToHost<const float*>(const float** d_list, const float** h_list, size_t byteCount);

    

    
    __host__ 
        void GPU_CUDA_convertWeightMatrix(float* d_list, size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Direction dir)
    {
        bool noHiddenLayer = !(hiddenY * hiddenX);

        if (noHiddenLayer)
        {
            // Transpose input Matrix
            if(dir == Direction::toDevice)
                kernel_transposeMatrix << <1, 1 >> > (d_list, inputCount, outputCount, _d_cudaInfo);
            else
                kernel_transposeMatrix << <1, 1 >> > (d_list, outputCount, inputCount, _d_cudaInfo);
        }
        else
        {
            // Transpose input Matrix
            if (dir == Direction::toDevice)
                kernel_transposeMatrix << <1, 1 >> > (d_list, inputCount, hiddenY, _d_cudaInfo);
            else
                kernel_transposeMatrix << <1, 1 >> > (d_list, hiddenY, inputCount, _d_cudaInfo);

            d_list += inputCount * hiddenY;

            // Transpose all Layers
            for (size_t l = 1; l < hiddenX; ++l)
            {
                kernel_transposeMatrix << <1, 1 >> > (d_list, hiddenY, _d_cudaInfo);
                d_list += hiddenY * hiddenY;
            }

            // Transpose output Matrix
            if (dir == Direction::toDevice)
                kernel_transposeMatrix << <1, 1 >> > (d_list, hiddenY, outputCount, _d_cudaInfo);
            else
                kernel_transposeMatrix << <1, 1 >> > (d_list, outputCount, hiddenY, _d_cudaInfo);
        }       
    }

    
    __host__ 
        void GPU_CUDA_learnBackpropagation(float* d_weights, float* d_deltaWeights, float* d_biasList, float* d_deltaBiasList, float* d_inputSignals, float* d_neuronOutputs, float* d_neuronNetinputs,
                                           size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, size_t neuronCount, size_t weightCount, Activation act,
                                           float* d_outputErrorList, float* d_expectedOutput, float learnParam)
    {
        float** dummyList;
        cudaMalloc(dummyList, 7 * sizeof(float*));
        cudaMemcpy(dummyList[0], d_deltaWeights, sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(dummyList[1], d_deltaBiasList, sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(dummyList[2], d_inputSignals, sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(dummyList[3], d_neuronOutputs, sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(dummyList[4], d_neuronNetinputs, sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(dummyList[5], d_outputErrorList, sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(dummyList[6], d_expectedOutput, sizeof(float*), cudaMemcpyHostToDevice);

        kernel_learnBackpropagationStream(d_weights, dummyList, d_biasList, dummyList+1, dummyList+2, dummyList + 3, dummyList + 4,
                                                     inputCount, hiddenX, hiddenY, outputCount, neuronCount, weightCount, act,
                                                     dummyList + 5, dummyList + 6, learnParam,1);
        cuda_handleError(cudaDeviceSynchronize());
        cudaFree(dummyList);
    }

    __host__ 
        void GPU_CUDA_learnBackpropagationStream(float* d_weights, float** d_deltaWeights, float* d_biasList, float** d_deltaBiasList, float** d_inputSignals, float** d_neuronOutputs, float** d_neuronNetinputs,
                                                 size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, size_t neuronCount, size_t weightCount, Activation act,
                                                 float** d_outputErrorList, float** d_expectedOutput, float learnParam, size_t streamSize)
    {
        kernel_learnBackpropagationStream(d_weights, d_deltaWeights, d_biasList, d_deltaBiasList, d_inputSignals, d_neuronOutputs, d_neuronNetinputs,
                                                     inputCount, hiddenX, hiddenY, outputCount, neuronCount, weightCount, act,
                                                     d_outputErrorList, d_expectedOutput, learnParam, streamSize);
        cuda_handleError(cudaDeviceSynchronize());
    }

    __host__ 
        void GPU_CUDA_learnBackpropagation_getOutputError(float* d_outputSignals, float* h_expectedOutputSignals,
                                                          float* h_outputErrors, size_t outputCount)
    {
        size_t maxThreadsPerBlock = 1024;

        float* d_exp;
        float* d_outErr;
        cudaMalloc(&d_exp, outputCount * sizeof(float));
        cudaMalloc(&d_outErr, outputCount * sizeof(float));
        cudaMemcpy(d_exp, h_expectedOutputSignals, outputCount * sizeof(float), cudaMemcpyHostToDevice);

        size_t blockSize = maxThreadsPerBlock;
        size_t numBlocks = (outputCount - 1) / blockSize + 1;
        kernel_learnBackpropagation_getOutputError << < numBlocks, blockSize >> >
            (d_outputSignals, d_exp, d_outErr, outputCount);
        cuda_handleError(cudaDeviceSynchronize());

        cudaMemcpy(h_outputErrors, d_outErr, outputCount * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_exp);
        cudaFree(d_outErr);
    }


    // Kernel functions

    __device__
        inline float kernel_net_activation_linear(float x)
    {
        return NET_ACTIVATION_LINEAR(x);
    }
    __device__
        inline float kernel_net_activation_finiteLinear(float x)
    {
        return NET_ACTIVATION_FINITELINEAR(x);
    }
    __device__
        inline float kernel_net_activation_gaussian(float x)
    {
        return NET_ACTIVATION_GAUSSIAN(x);
    }
    __device__
        inline float kernel_net_activation_sigmoid(float x)
    {
        return NET_ACTIVATION_SIGMOID(x);
    }
    __device__
        inline float kernel_net_activation_binary(float x)
    {
        return NET_ACTIVATION_BINARY(x);
    }

    __device__
        inline float kernel_net_activation_linear_derivetive(float x)
    {
        return NET_ACTIVATION_LINEAR_DERIVETIVE(x);
    }
    __device__
        inline float kernel_net_activation_finiteLinear_derivetive(float x)
    {
        return NET_ACTIVATION_FINITELINEAR_DERIVETIVE(x);
    }
    __device__
        inline float kernel_net_activation_gaussian_derivetive(float x)
    {
        return NET_ACTIVATION_GAUSSIAN_DERIVETIVE(x);
    }
    __device__
        inline float kernel_net_activation_sigmoid_derivetive(float x)
    {
        return NET_ACTIVATION_SIGMOID_DERIVETIVE(x);
    }

    __device__ 
    kernel_ActFp* kernel_net_getActivationFunction(Activation act)
    {
        switch (act)
        {
            default:
            case Activation::sigmoid:
                return &kernel_net_activation_sigmoid;
            case Activation::linear:
                return &kernel_net_activation_linear;
            case Activation::finiteLinear:
                return &kernel_net_activation_finiteLinear;
            case Activation::binary:
                return &kernel_net_activation_binary;
            case Activation::gauss:
                return &kernel_net_activation_gaussian;

        }
        return nullptr;
    }
    __device__ 
    kernel_ActFp* kernel_net_getActivationDerivetiveFunction(Activation act)
    {
        switch (act)
        {
            default:
            case Activation::sigmoid:
                return &kernel_net_activation_sigmoid_derivetive;
            case Activation::linear:
                return &kernel_net_activation_linear_derivetive;
            case Activation::finiteLinear:
                return &kernel_net_activation_finiteLinear_derivetive;
            case Activation::gauss:
                return &kernel_net_activation_gaussian_derivetive;

        }
        return nullptr;
    }

    __global__
        void kernel_net_calculateLayer(float* weights, float* biasList, float* inputSignals, 
                                       float* connectionSignalList, float* netinputList, float* neuronSignalList,
                                          size_t neuronCount, size_t inputSignalCount, kernel_ActFp* act)
    {
        // Get "Neuron index"
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (neuronCount == 0 || index >= neuronCount)
            return;
        float res = 0;
        const short tileSize = 32;
        // Declare shares memory for the signals
        __shared__ float sharedSignals[tileSize];
        size_t signalsBegin = 0;
        size_t signalsEnd   = 0;
        short tiles = inputSignalCount / tileSize + 1;
        short loadCount = tileSize / neuronCount + 1;
        short loadIndex = loadCount * threadIdx.x;

        // Split the neuron in segments for faster compute
        /*for (short tile = 0; tile <tiles; ++tile)
        {
            // Set new signal pointer (inputs of neuron)
            signalsBegin = signalsEnd;
            if (tile == tiles - 1)
                signalsEnd = inputSignalCount;
            else
                signalsEnd = (tile + 1) * tileSize;

            // Load signals from VRAM in shared memory
            // Load even distributed over all threads to maximize speed
            for (short i = 0; i < loadCount; ++i)
            {
                short signalIndex = i + loadIndex;

                if (signalsEnd > (signalIndex + signalsBegin))
                {
                    // Coalesced memory reading
                    sharedSignals[signalIndex] = inputSignals[signalIndex+signalsBegin];
                }
                // Sync all threads in this Block
                __syncthreads();
            }
            for (size_t i = signalsBegin; i < signalsEnd; ++i)
            {
                
                // Read weights
               // float weight = weights[index + inputSignalCount * i];
                float weight = weights[i * neuronCount + index];
                __syncthreads();
                
                // Ad the product of signal and weight
                res += weight * sharedSignals[i - signalsBegin];
            }
        }*/
        //double res_ = 0;
        for (size_t i = 0; i < inputSignalCount; ++i)
        //for (size_t i = 0; i < 1; ++i)
        {

            // Read weights
            size_t wIndex = index + neuronCount * i;
            float weight = weights[wIndex];
           
           // float weight = weights[i * neuronCount + index];

            // Ad the product of signal and weight
            //res_ += (double)weight * (double)inputSignals[i];
            // 
            // 
            // 
            //res += weight * inputSignals[i];
            float conSig = weight * inputSignals[i];
            connectionSignalList[wIndex] = conSig;
            res += conSig;
            



           /* float delta = (float)round((double)(weight * inputSignals[i] * 1000.l)) / 1000.f;
            
            uint32_t* valueBits = (uint32_t*)&delta;

            *valueBits = *valueBits & 0xffff0000;
            res += *(float*)valueBits;*/
            //res += inputSignals[i];
        } 

        //printf("%lf\n", res);
        // Set the resultat of the activationfunction of the netinput (res)
       // if (index == 3906)
        //    printf("%i %f\n", index,res);
        //res = (float)res_;
        res += biasList[index];
        res = round(res * 100.f) / 100.f;
        netinputList[index] = res;
        //if(index==0)
        //printf("G: %f\n", res);
        //outputSignals[index] = round(res * 100.f) / 100.f;
        neuronSignalList[index] = round((*act)(res) * 100.f) / 100.f;
    }

    __global__
        void kernel_calculateNet(float* weights, float* biasList, float** multiSignalVec, float** multiOutputVec, 
                                 float** multiConnectionSignalList, float** multiNetinputList, float** multiNeuronSignalList, size_t multiSignalSize,
                                 size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation act,
                                 CUDA_info* d_info)
    {
       
        kernel_ActFp* actPtr = kernel_net_getActivationFunction(act);
        
        size_t maxThreadsPerBlock = 256;
        if (d_info != nullptr)
        {
            maxThreadsPerBlock = d_info->maxThreadsPerBlock;
        }

        bool noHiddenLayer = !(hiddenY * hiddenX);

        /*// TEST
        for (size_t i = 0; i < multiSignalSize; ++i)
        {
            memset(multiNetinputList[i], 0, (hiddenX * hiddenY + outputCount) *sizeof(float));
            memset(multiNeuronSignalList[i], 0, (hiddenX * hiddenY + outputCount) *sizeof(float));
        }*/

        if (noHiddenLayer)
        {
            size_t blockSize = maxThreadsPerBlock;
            size_t numBlocks = (outputCount - 1) / blockSize + 1;
            for (size_t i = 0; i < multiSignalSize; ++i)
            {
                // Calculate the first Layer
                kernel_net_calculateLayer << < numBlocks, blockSize >> > (weights, biasList, multiSignalVec[i],
                                                                          multiConnectionSignalList[i], multiNetinputList[i], multiNeuronSignalList[i],
                                                                          outputCount, inputCount, actPtr);
            }
            for (size_t j = 0; j < multiSignalSize; ++j)
            {
                memcpy(multiOutputVec[j], multiNeuronSignalList[j], outputCount * sizeof(float));
            }
            
        }
        else
        {
            float* wStart = weights;
            size_t blockSize = maxThreadsPerBlock;
            size_t numBlocks = (hiddenY - 1) / blockSize + 1;
            float** tmpHiddenOutSignals1 = new float*[multiSignalSize];;
            //float** tmpHiddenOutSignals2 = new float*[multiSignalSize];
            size_t deltaWeightIndex = inputCount * hiddenY;
            for (size_t j = 0; j < multiSignalSize; ++j)
            {
                tmpHiddenOutSignals1[j] = multiNeuronSignalList[j];
                //tmpHiddenOutSignals2[j] = new float[hiddenY];

                // Calculate the first Layer
                kernel_net_calculateLayer << < numBlocks, blockSize >> > (weights, biasList, multiSignalVec[j], 
                                                                          multiConnectionSignalList[j], multiNetinputList[j], multiNeuronSignalList[j],
                                                                          hiddenY, inputCount, actPtr);
                multiConnectionSignalList[j] += deltaWeightIndex;
                multiNetinputList[j] += hiddenY;
                multiNeuronSignalList[j] += hiddenY;
            }
            
            // Increment the current start pos of the weights
            weights += deltaWeightIndex;
            biasList += hiddenY;

            // Wait until layer kernel is finished
            kernel_handleError(cudaDeviceSynchronize());
            deltaWeightIndex = hiddenY * hiddenY;
            for (size_t i = 1; i < hiddenX; ++i)
            {
                
                for (size_t j = 0; j < multiSignalSize; ++j)
                {
                    // Calculate all hidden Layers
                    
                    kernel_net_calculateLayer << < numBlocks, blockSize >> > (weights, biasList, tmpHiddenOutSignals1[j],  
                                                                              multiConnectionSignalList[j], multiNetinputList[j], multiNeuronSignalList[j],
                                                                              hiddenY, hiddenY, actPtr);
                    multiConnectionSignalList[j] += deltaWeightIndex;
                    tmpHiddenOutSignals1[j] = multiNeuronSignalList[j];
                    multiNetinputList[j] += hiddenY;
                    multiNeuronSignalList[j] += hiddenY;
                }
                // Increment the current start pos of the weights
                weights += deltaWeightIndex;
                biasList += hiddenY;

                // Swap the the signal lists: last outputs become now the new inputs
                //float** tmp = tmpHiddenOutSignals1;
                //tmpHiddenOutSignals1 = tmpHiddenOutSignals2;
                //tmpHiddenOutSignals2 = tmp;

                // Wait until layer kernel is finished
                kernel_handleError(cudaDeviceSynchronize());
            }

            numBlocks = (outputCount - 1) / blockSize + 1;
            for (size_t j = 0; j < multiSignalSize; ++j)
            {
                
                kernel_net_calculateLayer << < numBlocks, blockSize >> > (weights, biasList, tmpHiddenOutSignals1[j],  
                                                                          multiConnectionSignalList[j], multiNetinputList[j], multiNeuronSignalList[j],
                                                                          outputCount, hiddenY, actPtr);
            }

            // Wait until layer kernel is finished
            kernel_handleError(cudaDeviceSynchronize());

            /*for (size_t j = 0; j < multiSignalSize; ++j)
            {
                delete[] tmpHiddenOutSignals1[j];
                delete[] tmpHiddenOutSignals2[j]; 
            }*/
            delete[] tmpHiddenOutSignals1;
            //delete[] tmpHiddenOutSignals2;

            for (size_t j = 0; j < multiSignalSize; ++j)
            {
                size_t delta = hiddenY * hiddenX;;
                multiNetinputList[j] -= delta;
                memcpy(multiOutputVec[j], multiNeuronSignalList[j], outputCount * sizeof(float));
                multiNeuronSignalList[j] -= delta;
                multiConnectionSignalList[j] -= weights-wStart;
            }
        }
    }

    __global__
        void kernel_transposeMatrix(float* d_list, size_t width, CUDA_info* d_info)
    {
        size_t maxElement = kernel_gaussSum(width);

        size_t blockSize = 256;
        if(d_info != nullptr)
            d_info->maxThreadsPerBlock;
        size_t numBlocks = (maxElement - 1) / blockSize + 1;
        size_t sliceSize = 2048;
        if (numBlocks > sliceSize)
        {
            numBlocks = sliceSize;
        }

        for (size_t i = 0; i < width / sliceSize + 1; ++i)
        {
            kernel_transposeMatrix_internal << <numBlocks, blockSize >> > (d_list, width, maxElement, kernel_gaussSum(i * sliceSize));
        }
    }

    __global__
        void kernel_transposeMatrix_internal(float* d_list, size_t width, size_t maxIndex, size_t indexOffset)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x + indexOffset;
       // #define COALESCED
      
        size_t x = kernel_invGaussSum(index);
        size_t y = index - kernel_gaussSum(x);

        if (index >= maxIndex  || x == y)
        {
            return;
        }

        if (x > width || y > width)
            printf("ERROR: x>width || y>width %i\n",index);
        size_t elementIndex1 = y * width + x;
        size_t elementIndex2 = x * width + y;

#ifdef COALESCED
        //__shared__ float buffer1[1024];
        __shared__ float buffer2[1024];

        
        //buffer1[threadIdx.x] = d_list[elementIndex1];
        //__syncthreads();
        buffer2[threadIdx.x] = d_list[elementIndex2];
        __syncthreads();
        d_list[elementIndex1] = buffer2[threadIdx.x];
        __syncthreads();
        d_list[elementIndex2] = d_list[elementIndex1];

#else
        
        float tmp = d_list[elementIndex1];
        float tmp2 = d_list[elementIndex2];

        d_list[elementIndex1] = d_list[elementIndex2];
        __syncthreads();
        d_list[elementIndex2] = tmp;
        __syncthreads();
#endif
    }
    
    
    __global__
        void kernel_transposeMatrix(float* d_list, size_t width, size_t height, CUDA_info* d_info)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index > 0)
            return;

        size_t size = width * height;
        float* newList = new float[size];
        //for (size_t i = 0; i < size; ++i)
        //    newList[i] = 1.11;

        size_t maxThreadsPerBlock = 256;
        if (d_info != nullptr)
        {
            maxThreadsPerBlock = d_info->maxThreadsPerBlock;
        }

        dim3 numBlocks = dim3(width / sqrt((double)maxThreadsPerBlock) + 1, height / sqrt((double)maxThreadsPerBlock) + 1);
        dim3 numThreads = dim3(width / numBlocks.x+1, height / numBlocks.y+1);
        
        memcpy(newList, d_list, size * sizeof(float));
        kernel_transposeMatrix_rect_internal <<< numBlocks, numThreads >>> (d_list, newList, width, height);
        kernel_handleError(cudaDeviceSynchronize());
        
        delete[] newList;
    }



    __global__
        void kernel_transposeMatrix_rect_internal(float* d_list, float* tmpBuffer, size_t width, size_t height)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
            return;

      //  size_t inputIndex = y * width + x;
      //  size_t outputIndex = 


        size_t inputIndex = y * width + x;
       
        size_t outputIndex = x * height + y;
        d_list[outputIndex] = tmpBuffer[inputIndex];
       
    }

    __host__ 
        size_t gaussSum(size_t val)
    {
        return (val * (val + 1)) / 2;
    }
    
    __device__
        size_t kernel_gaussSum(size_t val)
    {
        return (val * (val + 1)) / 2;
    }

    __host__ 
        size_t invGaussSum(size_t sum)
    {
        return (size_t)floor((sqrt(8 * (double)sum + 1) - 1) / 2);
    }
    __device__
        size_t kernel_invGaussSum(size_t sum)
    {
        return (size_t)floor((sqrt(8 * (double)sum + 1) - 1) / 2);
    }

    __host__
        void kernel_learnBackpropagationStream(float* d_weights, float** d_deltaWeights, float* d_biasList, float** d_deltaBiasList, float** d_inputSignals, float** d_neuronOutputs, float** d_neuronNetinputs,
                                               size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, size_t neuronCount, size_t weightCount, Activation act,
                                               float** d_outputErrorList, float** d_expectedOutput, float learnParam, size_t streamSize)
    {
       //float** deltaW = new float*[streamSize];
       //float** deltaB = new float*[streamSize];

       //for (size_t i = 0; i < streamSize; ++i)
       //{
       //    //deltaW[i] = new float[weightCount];
       //    //deltaB[i] = new float[neuronCount];
       //    cudaMemset(d_deltaWeights[i], 0, weightCount * sizeof(float));
       //    cudaMemset(d_deltaBiasList[i], 0, neuronCount * sizeof(float));
       //}

        //kernel_ActFp* actDerivPtr; //= kernel_net_getActivationDerivetiveFunction(act);



        size_t blockSize = 1;
        size_t numBlocks = (streamSize - 1) / blockSize + 1;

            kernel_learnBackpropagation<<< numBlocks, blockSize>>>
                (d_weights, d_deltaWeights, d_biasList, d_deltaBiasList,
                 d_inputSignals, d_neuronOutputs, d_neuronNetinputs,
                 inputCount, hiddenX, hiddenY, outputCount, neuronCount, weightCount, act,
                 d_outputErrorList, d_expectedOutput, streamSize);


        cuda_handleError(cudaGetLastError());
        cuda_handleError(cudaDeviceSynchronize());

        numBlocks = (weightCount - 1) / blockSize + 1;
        kernel_learnBackpropagation_applyDeltaValue << < numBlocks, blockSize >> > (d_weights, d_deltaWeights, learnParam, streamSize, weightCount);
        cuda_handleError(cudaGetLastError());

        numBlocks = (neuronCount - 1) / blockSize + 1;
        kernel_learnBackpropagation_applyDeltaValue << < numBlocks, blockSize >> > (d_biasList, d_deltaBiasList, learnParam, streamSize, neuronCount);
        cuda_handleError(cudaDeviceSynchronize());

       // for (size_t i = 0; i < streamSize; ++i)
       // {
       //     delete[] deltaW[i];
       //     delete[] deltaB[i];
       // }
       // delete[] deltaW;
       // delete[] deltaB;
    }
    __global__ 
        void kernel_learnBackpropagation(float* d_weights, float** d_deltaWeights, float* d_biasList, float** d_deltaBiasList, 
                                         float** d_inputSignals, float** d_neuronOutputs, float** d_neuronNetinputs,
                                         size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, size_t neuronCount, size_t weightCount, Activation act,//kernel_ActFp* actDerivPtr,
                                         float** d_outputErrorList, float** d_expectedOutput, size_t streamSize)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= streamSize)
            return;
        kernel_ActFp* actDerivPtr = kernel_net_getActivationDerivetiveFunction(act);

        float* dWeights = d_deltaWeights[index];
        float* dBias    = d_deltaBiasList[index];
        float* inp      = d_inputSignals[index];
        float* out      = d_neuronOutputs[index];
        float* net      = d_neuronNetinputs[index];
        float* outErr   = d_outputErrorList[index];
        float* expOut   = d_expectedOutput[index];

        memset(dWeights, 0, weightCount * sizeof(float));
        memset(dBias, 0, neuronCount * sizeof(float));

        //kernel_ActFp* actDerivPtr = kernel_net_getActivationDerivetiveFunction(act);
        size_t outputNeuronBeginIndex = neuronCount - outputCount - 1;
        float *outputError = new float[outputCount];
        //float *outputDifference = new float[outputCount];

        size_t maxThreadsPerBlock = 1024;

        size_t blockSize = maxThreadsPerBlock;
        size_t numBlocks = (outputCount - 1) / blockSize + 1;
        kernel_learnBackpropagation_getOutputError<<< numBlocks, blockSize>>>
            (out + neuronCount - outputCount, expOut,
             outErr, outputCount);
        kernel_handleError(cudaDeviceSynchronize());


        for (size_t y = 0; y < outputCount; ++y)
        {
            float netinput = net[outputNeuronBeginIndex + y];
            float derivetive = (*actDerivPtr)(netinput);
            outputError[y] = derivetive * outErr[y];

            float deltaBias = outputError[y];
            dBias[outputNeuronBeginIndex + y] += deltaBias;
        }


        // Calculate errors for each layer:
        if (hiddenX > 0)
        {
            float* nextHiddenError = nullptr;


            for (long long x = hiddenX - 1; x >= 0; --x)
            {
                size_t hiddenNeuronBeginIndex = x * hiddenY;
                size_t weightBeginIndex = inputCount * hiddenY + x * hiddenY * hiddenY;
                float* hiddenError = new float[hiddenY];
                

                float* tmp_weights = d_weights + weightBeginIndex;
                float* tmp_deltaWeights = dWeights + weightBeginIndex;
                float* tmp_deltaBias = dBias + hiddenNeuronBeginIndex;
                float* tmp_neuronOut = out + hiddenNeuronBeginIndex;
                float* tmp_neuronNetinput = net + hiddenNeuronBeginIndex;
                float* tmp_IError = hiddenError;
                float* tmp_JError = nextHiddenError;
                size_t layerISize = hiddenY;
                size_t layerJSize = hiddenY;
                if (x == hiddenX - 1)
                {
                    tmp_JError = outputError;
                    layerJSize = outputCount;
                }
                

                size_t blockSize = maxThreadsPerBlock;
                size_t numBlocks = (hiddenY - 1) / blockSize + 1;
                kernel_learnBackpropagation_calculateLayerDeltaW << < numBlocks, blockSize >> > (tmp_weights, tmp_deltaWeights, tmp_deltaBias,
                                                                                                 tmp_neuronOut, tmp_neuronNetinput,
                                                                                                 tmp_IError, tmp_JError,
                                                                                                 actDerivPtr, layerISize, layerJSize);
                kernel_handleError(cudaDeviceSynchronize());
                /*
                for (size_t y = 0; y < hiddenY; ++y)
                {
                    float sumNextLayerErrors = 0;
                    
                    if (x == hiddenX - 1)
                    {
                        size_t weightIndex = inputCount * hiddenY + x * hiddenY * hiddenY + y * outputCount;
                        // Calculate the errorsum of the outputLayer			
                        for (size_t i = 0; i < outputCount; ++i)
                        {
                            sumNextLayerErrors += outputError[i] * d_weights[weightIndex + i];

                            // Change the weight
                            float deltaW = d_neuronOutputs[hiddenNeuronBeginIndex + y] * outputError[i];
                            d_deltaWeights[weightIndex + i] += deltaW;
                        }

                    }
                    else
                    {
                        size_t weightIndex = inputCount * hiddenY + x * hiddenY * hiddenY + y * hiddenY;
                        // Calculate the errorsum of the hiddenLayer
                        for (size_t i = 0; i < hiddenY; ++i)
                        {
                            sumNextLayerErrors += nextHiddenError[i] * d_weights[weightIndex + i];

                            // Change the weight
                            float deltaW = d_neuronOutputs[hiddenNeuronBeginIndex + y] * nextHiddenError[i];
                            d_deltaWeights[weightIndex + i] += deltaW;
                        }
                    }

                    hiddenError[y] = (*actDerivPtr)(d_neuronNetinputs[hiddenNeuronBeginIndex + y]) *
                        sumNextLayerErrors;

                    float deltaBias = hiddenError[y];
                    d_deltaBiasList[x * hiddenY + y] += deltaBias;
                }*/
                if (nextHiddenError)
                    delete nextHiddenError;
                nextHiddenError = hiddenError;

                if (x == 0)
                {
                    // Change the last weights: inputweights
                    for (size_t y = 0; y < inputCount; ++y)
                    {
                        for (size_t i = 0; i < hiddenY; ++i)
                        {
                            // Change the weight
                            float deltaW = inp[y] * hiddenError[i];
                            dWeights[y * hiddenY + i] += deltaW;
                        }
                    }
                }
            }
            delete nextHiddenError;
        }
        else
        {
            // Only one Layer: outputLayer

            // Change the last waights: inputweights
            for (size_t y = 0; y < inputCount; ++y)
            {
                for (size_t i = 0; i < outputCount; ++i)
                {
                    // Change the weight
                    float deltaW = inp[y] * outputError[i];
                    dWeights[y * outputCount + i] += deltaW;
                }
            }
        }
        delete[] outputError;
       // delete[] outputDifference;
    }

    __global__
        void  kernel_learnBackpropagation_applyDeltaValue(float* d_originalList, float** d_deltaList, float factor, size_t listSize, size_t cout)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cout)
            return;

        for(size_t i=0; i< listSize; ++i)
            d_originalList[index] += d_deltaList[i][index] * factor;
    }


    // Calculates the Error of the output layer
    __global__
        void kernel_learnBackpropagation_getOutputError(float* d_outputSignals, float* d_expectedOutputSignals,
                                                        float* d_outputErrors, size_t outputCount)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= outputCount)
            return;

 
        float expected = d_expectedOutputSignals[index];
        float output = d_outputSignals[index];
        float difference = (expected - output);
        d_outputErrors[index] = difference;
    }

    __global__ 
        void kernel_learnBackpropagation_calculateLayerDeltaW(float* d_weights, float* d_deltaW, float* d_deltaB, 
                                                              float* d_neuronOutputs, float* d_netinputs,
                                                              float* d_layerIErrorList, float* d_LayerJErrorList,
                                                              kernel_ActFp* actDerivPtr, size_t layerIY, size_t layerJY)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= layerIY)
            return;
			
        float sumNextLayerErrors = 0;
        size_t weightIndex = index * layerJY;
        for (size_t i = 0; i < layerJY; ++i)
        {
            sumNextLayerErrors += d_LayerJErrorList[i] * d_weights[weightIndex + i];

            // Change the weight
            float deltaW = d_neuronOutputs[index] * d_LayerJErrorList[i];
            d_deltaW[weightIndex + i] += deltaW;
        }
        d_layerIErrorList[index] = (*actDerivPtr)(d_netinputs[index]) *
            sumNextLayerErrors;

        float deltaBias = d_layerIErrorList[index];
        d_deltaB[index] += deltaBias;

        
        /*
        // Hat einen unbekannten Fehler, tritt auf bei grossen Netzen
        __shared__ float layerJErrorList[1024];

        float sumNextLayerErrors = 0;
        size_t weightIndex = index * layerJY;
        size_t iterations = layerJY / 1024 + 1;
        for (size_t it = 0; it < iterations; ++it)
        {
            size_t id = index + it * 1024;
            if(id < layerJY)
                layerJErrorList[threadIdx.x] = d_LayerJErrorList[id];
            __syncthreads();
            if (index < layerIY)
            {
                size_t iCount = layerJY % 1024;
                size_t wOffset = weightIndex + it * 1024;
                for (size_t i = 0; i < iCount; ++i)
                {

                    sumNextLayerErrors += layerJErrorList[i] * d_weights[weightIndex + wOffset];

                    // Change the weight
                    float deltaW = d_neuronOutputs[index] * layerJErrorList[i];
                    d_deltaW[weightIndex + wOffset] += deltaW;
                    ++wOffset;
                }
            }
        }

        if (index < layerIY)
        {
            d_layerIErrorList[index] = (*actDerivPtr)(d_netinputs[index]) *
                sumNextLayerErrors;

            float deltaBias = d_layerIErrorList[index];
            d_deltaB[index] += deltaBias;
        }*/
        
    }

    /*
    __device__ 
        void kernel_calculateOutputError(float** d_netinpuitMultiSignals, float** d_outputMultiSignals, float** d_expectedOutputMultiSignal,
                                         float** d_errorMultiList, kernel_ActFp* derivetiveFunc, 
                                         size_t outputNeuronStartIndex, size_t  outputCount, size_t signalCount)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index > outputCount)
            return;

        size_t outputNeuronIndex = index + outputNeuronStartIndex;

        // Error = f'(netinput) * (expected - output)
        for (size_t i = 0; i < signalCount; ++i)
        {
            d_errorMultiList[i][outputNeuronIndex] = (*derivetiveFunc)(d_netinpuitMultiSignals[i][outputNeuronIndex]) *
                                         (d_expectedOutputMultiSignal[i][index] - d_outputMultiSignals[i][index]);
        }
    }

    #define CUDA_CALCULATE_HIDDEN_ERROR_SLICE_SIZE 32
    __device__ 
        void kernel_calculateHiddenError(float** d_netinpuitMultiSignals, float* d_weightList,
                                         float** d_errorMultiList, kernel_ActFp* derivetiveFunc,
                                         size_t hiddenNeuronStartIndex, size_t iNeuronYCount, 
                                         size_t jNeuronYCount, size_t signalCount)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index > neuronYCount)
            return;
        size_t hiddenNeuronIndex = index + hiddenNeuronStartIndex;
        size_t nextLayerNeuronStartIndex = hiddenNeuronStartIndex + iNeuronYCount;
        
       // __shared__ float jErrorList[CUDA_CALCULATE_HIDDEN_ERROR_SLICE_SIZE];

        for (size_t i = 0; i < signalCount; ++i)
        {
            float sumWeightedError = 0;
            for (size_t j = 0; j < jNeuronYCount; ++j)
            {
                sumWeightedError += d_errorMultiList[i][nextLayerNeuronStartIndex + j] * d_weightList[j];
            }
            d_errorMultiList[i][hiddenNeuronIndex] = (*derivetiveFunc)(d_netinpuitMultiSignals[i][hiddenNeuronIndex]) * sumWeightedError;
        }

    }

    #define CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE 32
  //  #define CUDA_CHANGE_LAYER_WEIGHTS_ITERATION_SIZE 
    __device__ 
        void kernel_changeLayerWeights(float* d_weightList, float** d_neuronMultiSignals, float** d_errorMultiList,
                                       size_t neuronCountI, size_t neuronCountJ, size_t signalCount, float learnRate)
    {
        dim3 numBlocks = dim3(neuronCountI / CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE + 1, 
                              neuronCountJ / CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE + 1);
        dim3 numThreads = dim3(CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE,
                               CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE);

        size_t weightCount = neuronCountI * neuronCountJ;
        float* deltaW = new float[weightCount];

        kernel_changeLayerWeights_slice << < numBlocks, numThreads >> > (deltaW, d_neuronMultiSignals, d_errorMultiList,
                                                                         neuronCountI, neuronCountJ, signalCount, 0, signalCount);

        cudaDeviceSynchronize();
        numBlocks.x = weightCount / 1024 + 1;
        numBlocks.y = 0;
        numThreads.x = 1024;
        numThreads.y = 0;
        kernel_applyDeltaWeight << < numBlocks, numThreads >> > (deltaW, d_weightList);
    }
    __device__ 
        void kernel_changeLayerWeights_slice(float* d_deltaW, float** d_neuronMultiSignals, float** d_errorMultiList,
                                             size_t neuronCountI, size_t neuronCountJ, size_t signalCount,
                                             size_t iteration, size_t iterationSize)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= neuronCountJ || y >= neuronCountI)
            return;
        size_t iterationOffset = iteration * iterationSize;
        __shared__ float outputSignals[CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE][CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE];
        __shared__ float errorValues  [CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE][CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE];
        __shared__ float result[CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE][CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE];

        outputSignals[threadIdx.x][threadIdx.y] = d_neuronMultiSignals[iterationOffset + x][y];
        errorValues  [threadIdx.x][threadIdx.y] = d_errorMultiList    [x][iterationOffset + y];

        result[threadIdx.x][threadIdx.y] = 0;
        __syncthreads();

        for (size_t i = 0; i < CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE; ++i)
        {
            result[threadIdx.x][threadIdx.y] += outputSignals[i][threadIdx.y] * errorValues[threadIdx.x][i];
        }
        __syncthreads();

        d_deltaW[threadIdx.x + threadIdx.y * CUDA_CHANGE_LAYER_WEIGHTS_SLICE_SIZE] += result[threadIdx.x][threadIdx.y];

    }
    __device__ 
        void kernel_applyDeltaWeight(float* d_deltaW, float* d_weights, size_t size)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= size)
            return;
        d_weights[index] += d_deltaW[index];
    }*/

    __global__
        void kernel_offsetScale(float *d_list, float offset, float scale, size_t size, CUDA_info* d_info)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        
        // size may differ on other devices
        __shared__ float buffer[1024];
        if (index >= size)
            return;

        buffer[threadIdx.x] = d_list[index];
        buffer[threadIdx.x] = (buffer[threadIdx.x] + offset) * scale;
        __syncthreads();
        d_list[index] = buffer[threadIdx.x];
    }
    __host__ void cuda_handleError(cudaError_t err)
    {
        switch (err)
        {
            case cudaError_t::cudaSuccess:
                return;
            default:
            {
                printf("CudaError: %i\n",err);
            }
        }
    }
    __device__ void kernel_handleError(cudaError_t err)
    {
        switch (err)
        {
            case cudaError_t::cudaSuccess:
                return;
            default:
            {
                printf("CudaError: %i\n",err);
            }
        }
    }
}

