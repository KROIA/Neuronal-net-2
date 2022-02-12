#include "../inc/net_kernel.cuh"


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
        float* matrix = new float[w * h];
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
        cudaDeviceSynchronize();

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
            _h_cudaInfo = new CUDA_info;
            _h_cudaInfo->maxThreadDim.x = h_deviceProp.maxThreadsDim[0];
            _h_cudaInfo->maxThreadDim.y = h_deviceProp.maxThreadsDim[1];
            _h_cudaInfo->maxThreadDim.z = h_deviceProp.maxThreadsDim[2];
            _h_cudaInfo->maxThreadsPerBlock = h_deviceProp.maxThreadsPerBlock;
            _h_cudaInfo->totalGlobalMemory = h_deviceProp.totalGlobalMem;
        }
        return h_deviceProp;
    }
    __host__ 
        void GPU_CUDA_calculateNet(float* weights, float** multiSignalVec, float** multiOutputVec, size_t multiSignalSize,
                                   size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation activation,
                                   CUDA_info* d_info)
    {

        kernel_calculateNet << <1, 1 >> > (weights, multiSignalVec, multiOutputVec, multiSignalSize,
                                           inputCount, hiddenX, hiddenY, outputCount, activation,
                                           d_info);
        cudaDeviceSynchronize();
    }
    __host__
        void GPU_CUDA_getRandomWeight(float min, float max, float* h_list, size_t elements)
    {
        size_t blockSize = _h_cudaInfo->maxThreadsPerBlock;
        size_t numBlocks = (elements - 1) / blockSize + 1;
        curandGenerator_t gen;
        float* d_list;
        cudaMalloc(&d_list, elements * sizeof(float));
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_list, elements);
        cudaDeviceSynchronize();
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
    __host__ void GPU_CUDA_transferToDevice(T* d_list, T* h_list, size_t byteCount)
    {
        cuda_handleError(cudaMemcpy(d_list, h_list, byteCount, cudaMemcpyHostToDevice));
    }
    template NET_API __host__ void GPU_CUDA_transferToDevice<float>(float* d_list, float* h_list, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_transferToDevice<float*>(float** d_list, float** h_list, size_t byteCount);

    template <typename T>
    __host__ void GPU_CUDA_transferToHost(T* d_list, T* h_list, size_t byteCount)
    {
        cuda_handleError(cudaMemcpy(h_list, d_list, byteCount, cudaMemcpyDeviceToHost));
    }
    template NET_API __host__ void GPU_CUDA_transferToHost<float>(float* d_list, float* h_list, size_t byteCount);
    template NET_API __host__ void GPU_CUDA_transferToHost<float*>(float** d_list, float** h_list, size_t byteCount);

    

    
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

    
    


    // Kernel functions

    __device__
        inline float kernel_net_activation_linear(float x)
    {
        return NET_ACTIVATION_LINEAR(x);
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
    kernel_ActFp* kernel_net_getActivationFunction(Activation act)
    {
        switch (act)
        {
            default:
            case Activation::sigmoid:
                return &kernel_net_activation_sigmoid;
            case Activation::linear:
                return &kernel_net_activation_linear;
            case Activation::gauss:
                return &kernel_net_activation_gaussian;

        }
    }

    __global__
        void kernel_net_calculateLayer(float* weights, float* inputSignals, float* outputSignals,
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
        for (short tile = 0; tile <tiles; ++tile)
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
        }
        // Set the resultat of the activationfunction of the netinput (res)
        outputSignals[index] = (*act)(res);
    }

    __global__
        void kernel_calculateNet(float* weights, float** multiSignalVec, float** multiOutputVec, size_t multiSignalSize,
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

        if (noHiddenLayer)
        {
            size_t blockSize = maxThreadsPerBlock;
            size_t numBlocks = (outputCount - 1) / blockSize + 1;
            for (size_t i = 0; i < multiSignalSize; ++i)
            {
                // Calculate the first Layer
                kernel_net_calculateLayer << < numBlocks, blockSize >> > (weights, multiSignalVec[i], multiOutputVec[i], outputCount, inputCount, actPtr);
            }
            
        }
        else
        {
            size_t blockSize = maxThreadsPerBlock;
            size_t numBlocks = (hiddenY - 1) / blockSize + 1;
            float** tmpHiddenOutSignals1 = new float*[multiSignalSize];
            float** tmpHiddenOutSignals2 = new float*[multiSignalSize];
            for (size_t j = 0; j < multiSignalSize; ++j)
            {
                tmpHiddenOutSignals1[j] = new float[hiddenY];
                tmpHiddenOutSignals2[j] = new float[hiddenY];

                // Calculate the first Layer
                kernel_net_calculateLayer << < numBlocks, blockSize >> > (weights, multiSignalVec[j], tmpHiddenOutSignals1[j], hiddenY, inputCount, actPtr);
            }
            
            // Increment the current start pos of the weights
            weights += inputCount * hiddenY;

            // Wait until layer kernel is finished
            cudaDeviceSynchronize();

            for (size_t i = 1; i < hiddenX; ++i)
            {
                for (size_t j = 0; j < multiSignalSize; ++j)
                {
                    // Calculate all hidden Layers
                    kernel_net_calculateLayer << < numBlocks, blockSize >> > (weights, tmpHiddenOutSignals1[j], tmpHiddenOutSignals2[j], hiddenY, hiddenY, actPtr);
                }
                // Increment the current start pos of the weights
                weights += hiddenY * hiddenY;

                // Swap the the signal lists: last outputs become now the new inputs
                float** tmp = tmpHiddenOutSignals1;
                tmpHiddenOutSignals1 = tmpHiddenOutSignals2;
                tmpHiddenOutSignals2 = tmp;

                // Wait until layer kernel is finished
                cudaDeviceSynchronize();
            }

            numBlocks = (outputCount - 1) / blockSize + 1;
            for (size_t j = 0; j < multiSignalSize; ++j)
            {
                kernel_net_calculateLayer << < numBlocks, blockSize >> > (weights, tmpHiddenOutSignals1[j], multiOutputVec[j], outputCount, hiddenY, actPtr);
            }

            // Wait until layer kernel is finished
            cudaDeviceSynchronize();

            for (size_t j = 0; j < multiSignalSize; ++j)
            {
                delete[] tmpHiddenOutSignals1[j];
                delete[] tmpHiddenOutSignals2[j]; 
            }
            delete[] tmpHiddenOutSignals1;
            delete[] tmpHiddenOutSignals2;
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
        for (size_t i = 0; i < size; ++i)
            newList[i] = 1.11;

        size_t maxThreadsPerBlock = 256;
        if (d_info != nullptr)
        {
            maxThreadsPerBlock = d_info->maxThreadsPerBlock;
        }

        dim3 numBlocks = dim3(width / sqrt((double)maxThreadsPerBlock) + 1, height / sqrt((double)maxThreadsPerBlock) + 1);
        dim3  numThreads = dim3(width / numBlocks.x, height / numBlocks.y);
        
        memcpy(newList, d_list, size * sizeof(float));
        kernel_transposeMatrix_rect_internal <<< numBlocks, numThreads >>> (d_list, newList, width, height);
        cudaDeviceSynchronize();
        
        delete[] newList;
    }



    __global__
        void kernel_transposeMatrix_rect_internal(float* d_list, float* tmpBuffer, size_t width, size_t height)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
            return;

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


    __host__ void cuda_handleError(cudaError_t err)
    {
        switch (err)
        {
            case cudaError_t::cudaSuccess:
                return;
            default:
            {
                std::cout << "CudaError: " << err << "\n";
            }
        }
    }
}

