#include "../inc/net_kernel.cuh"

#define BLOCK_DIM 16

namespace NeuronalNet 
{
    __global__ void transpose(float* odata, float* idata, int width, int height)
    {
        __shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

        // read the matrix tile into shared memory
            // load one element per thread from device memory (idata) and store it
            // in transposed order in block[][]
        unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
        unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
        if ((xIndex < width) && (yIndex < height))
        {
            unsigned int index_in = yIndex * width + xIndex;
            block[threadIdx.y][threadIdx.x] = idata[index_in];
        }

        // synchronise to ensure all writes to block[][] have completed
        __syncthreads();

        // write the transposed matrix tile to global memory (odata) in linear order
        xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
        yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
        if ((xIndex < height) && (yIndex < width))
        {
            unsigned int index_out = yIndex * height + xIndex;
            odata[index_out] = block[threadIdx.x][threadIdx.y];
        }
    }
    __host__ 
        double GPU_CUDA_transposeMatrix(float* d_list, size_t width)
    {/*
        size_t maxElement = gaussSum(width);
        size_t elementCount = width * width;

        float* d_list1;
        float* d_list2;
        GPU_CUDA_allocMem(d_list1, elementCount * sizeof(float));
        GPU_CUDA_allocMem(d_list2, elementCount * sizeof(float));
        GPU_CUDA_transferToDevice(d_list1, h_list, elementCount * sizeof(float));

        dim3 grid(width / BLOCK_DIM, width / BLOCK_DIM, 1);
        dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
        int blockSize = GPU_CUDA_getSpecs().maxThreadsPerBlock;
        int numBlocks = (maxElement - 1) / blockSize + 1;
        auto t1 = std::chrono::high_resolution_clock::now();
        transpose <<<grid, threads >>> (d_list2,d_list1, width, width);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        double transposeTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double transposeTimeMs = transposeTimeNs / 1000000;
        std::cout << "Transposetime: " << transposeTimeNs << " ns = " << transposeTimeMs << " ms\n";
        GPU_CUDA_transferToHost(d_list2, h_list, elementCount * sizeof(float));
        GPU_CUDA_freeMem(d_list1);
        GPU_CUDA_freeMem(d_list2);
        return transposeTimeMs;
        */
        size_t maxElement = gaussSum(width);
      //  size_t elementCount = width * width;

       // float* d_list;
       // GPU_CUDA_allocMem(d_list, elementCount * sizeof(float));
       // GPU_CUDA_transferToDevice(d_list, h_list, elementCount * sizeof(float));


        int blockSize = GPU_CUDA_getSpecs().maxThreadsPerBlock;
        int numBlocks = (maxElement - 1) / blockSize + 1;
        size_t sliceSize = 2048 / 4;
        if (numBlocks > sliceSize)
        {
            //std::cout << "numBlocks > 2048\n";
            numBlocks = sliceSize;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < width / sliceSize + 1; ++i)
        {
            kernel_transposeMatrix << <numBlocks, blockSize >> > (d_list, width, maxElement, gaussSum(i * sliceSize));
            
        }
        //kernel_transposeMatrix << <numBlocks, blockSize >> > (d_list, width, maxElement, 0);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        double transposeTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        double transposeTimeMs = transposeTimeNs / 1000000;
        //std::cout << "Transposetime: " << transposeTimeNs <<" ns = " << transposeTimeMs << " ms\n";
      //  GPU_CUDA_transferToHost(d_list, h_list, elementCount * sizeof(float));
      //  GPU_CUDA_freeMem(d_list);
        return transposeTimeMs;
    }
    __host__
        double GPU_CUDA_transposeMatrix2(float* d_list1, float* d_list2, size_t width)
    {
        size_t maxElement = gaussSum(width);
        size_t elementCount = width * width;

      // float* d_list1;
      // float* d_list2;
      // GPU_CUDA_allocMem(d_list1, elementCount * sizeof(float));
      // GPU_CUDA_allocMem(d_list2, elementCount * sizeof(float));
      // GPU_CUDA_transferToDevice(d_list1, h_list, elementCount * sizeof(float));

        dim3 grid(width / BLOCK_DIM, width / BLOCK_DIM, 1);
        dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
        int blockSize = GPU_CUDA_getSpecs().maxThreadsPerBlock;
        int numBlocks = (maxElement - 1) / blockSize + 1;
        auto t1 = std::chrono::high_resolution_clock::now();
        transpose <<<grid, threads >>> (d_list2,d_list1, width, width);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        double transposeTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        double transposeTimeMs = transposeTimeNs / 1000000;
        //std::cout << "Transposetime: " << transposeTimeNs << " ns = " << transposeTimeMs << " ms\n";
      //  GPU_CUDA_transferToHost(d_list2, h_list, elementCount * sizeof(float));
      //  GPU_CUDA_freeMem(d_list1);
      //  GPU_CUDA_freeMem(d_list2);
        return transposeTimeMs;
    }


    __host__ 
        cudaDeviceProp GPU_CUDA_getSpecs()
    {
        cudaDeviceProp h_deviceProp;
        cudaGetDeviceProperties(&h_deviceProp, 0);
        return h_deviceProp;
    }
    __host__ 
        void GPU_CUDA_calculateNet(float* weights, float* signals, float* outpuSignals,
                               size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation activation)
    {
        nvtxRangePush(__FUNCTION__);
        nvtxMark("Waiting...");
        kernel_calculateNet << <1, 1 >> > (weights, signals, outpuSignals,
                                           inputCount, hiddenX, hiddenY, outputCount, activation);
        cudaDeviceSynchronize();
        nvtxRangePop();
    }
    __host__
        void GPU_CUDA_getRandomWeight(float min, float max, float* h_list, size_t elements)
    {
        /*curandStatus* d_state;
        size_t maxThreadPerBlock = 1024;
        size_t blockSize = maxThreadPerBlock;
        size_t numBlocks = (elements - 1) / blockSize + 1;
        float* d_list;
        cudaMalloc(&d_state, blockSize * numBlocks);
        cudaMalloc(&d_list, elements * sizeof(float));
        kernel_randomInit <<<numBlocks, blockSize >>> (d_state);
        kernel_getRandomWeight <<<numBlocks, blockSize >>> (min,max,d_state, d_list, elements);
        cudaFree(d_state);
        cudaFree(d_list);*/


        size_t maxThreadPerBlock = 1024;
        size_t blockSize = maxThreadPerBlock;
        size_t numBlocks = (elements - 1) / blockSize + 1;
        curandGenerator_t gen;
        float* d_list;
        cudaMalloc(&d_list, elements * sizeof(float));
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_list, elements);
        cudaDeviceSynchronize();
        //kernel_scaleRandomWeight <<<numBlocks, blockSize >>> (min, max, d_list, elements);
        //cudaDeviceSynchronize();
        cudaMemcpy(h_list, d_list, elements * sizeof(float), cudaMemcpyDeviceToHost);
       /* for (size_t i = 0; i < 100; ++i)
        {
            printf("%f\n", h_list[i]);
        }*/
        cudaFree(d_list);
    }


    __host__
        void GPU_CUDA_allocMem(float*& d_list, size_t byteCount)
    {
        d_list = nullptr;
        cuda_handleError(cudaMalloc(&d_list, byteCount));
    }

    __host__ 
        void GPU_CUDA_freeMem(float*& d_list)
    {
        if (!d_list)
            return;
        cudaError_t err = cudaFree(d_list);
        cuda_handleError(err);
        if(err == cudaError::cudaSuccess)
            d_list = nullptr;
    }

    __host__
        void GPU_CUDA_transferToDevice(float* d_list, float* h_list, size_t byteCount)
    {
        cuda_handleError(cudaMemcpy(d_list, h_list, byteCount, cudaMemcpyHostToDevice));
    }
    __host__
        void GPU_CUDA_transferToHost(float* d_list, float* h_list, size_t byteCount)
    {
       cuda_handleError(cudaMemcpy(h_list, d_list, byteCount, cudaMemcpyDeviceToHost));
    }
    __host__ 
        void GPU_CUDA_convertWeightToGPUWeight(float* d_list, size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount)
    {

    }


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
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        
        

        
        if (neuronCount == 0 || index >= neuronCount)
            return;

      
        

        float res = 0;
        //weights += index * inputSignalCount;

       // ramStorage = (Storage*)weights;
       // storage = *ramStorage;
       
       
       
       /*
        for (size_t i = 0; i < inputSignalCount; ++i)
        {
            res += weights[i] * inputSignals[i];
        }*/

        const short tileSize = 32;
        __shared__ float sharedSignals[tileSize];
        
        size_t signalsBegin = 0;
        size_t signalsEnd   = 0;
        short tiles = inputSignalCount / tileSize + 1;
        short loadCount = tileSize / neuronCount + 1;
        short loadIndex = loadCount * threadIdx.x;


        for (short tile = 0; tile <tiles; ++tile)
        {
            signalsBegin = signalsEnd;
            if (tile == tiles - 1)
                signalsEnd = inputSignalCount;
            else
                signalsEnd = (tile + 1) * tileSize;


            
            for (short i = 0; i < loadCount; ++i)
            {
                short signalIndex = i + loadIndex;

                if (signalsEnd > (signalIndex + signalsBegin))
                {
                     sharedSignals[signalIndex] = inputSignals[signalIndex+signalsBegin];
                }
                __syncthreads();
            }


            for (size_t i = signalsBegin; i < signalsEnd; ++i)
            {
                //float weight = weights[index * (i+1)];


               // res += *((float*)(&storage) + storageCount) * sharedSignals[i - signalsBegin];
                
                
                float weight = weights[index + inputSignalCount * i];
                //float weight = weights[index * inputSignalCount + i];
                __syncthreads();
                res += weight * sharedSignals[i - signalsBegin];
                
                //res += sharedSignals[i - signalsBegin];

            //   if ((++storageCount) >= 1)
            //   {
            //       storageCount = 0;
            //
            //       if ((i + 8) < signalsEnd)
            //       {
            //
            //           storage = *ramStorage;
            //           ++ramStorage;
            //       }
            //       else
            //       {
            //           /*for (size_t j = i; j < signalsEnd; ++j)
            //           {
            //               *((float*)&storage + j) = weights[j];
            //           }*/
            //       }
            //       
            //   }
            }
        }

        //__syncthreads();
        outputSignals[index] = (*act)(res);
    }

    __global__
        void kernel_calculateNet(float* weights, float* signals, float* outpuSignals,
                        size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation act)
    {
        
        kernel_ActFp* actPtr = kernel_net_getActivationFunction(act);
        
        size_t maxThreadPerBlock = 1024;

        size_t blockSize = maxThreadPerBlock;
        size_t numBlocks = (hiddenY - 1) / blockSize + 1;
        float* tmpHiddenOutSignals1 = new float[hiddenY];
        float* tmpHiddenOutSignals2 = new float[hiddenY];
        cudaDeviceSynchronize();
        //kernel_net_calculateLayer <<< numBlocks, blockSize >>> (weights, signals, tmpHiddenOutSignals1, hiddenY, inputCount, actPtr);
        kernel_net_calculateLayer <<< 1, 1 >>> (weights, signals, tmpHiddenOutSignals1, hiddenY, inputCount, actPtr);
        weights += inputCount * hiddenY;
        cudaDeviceSynchronize();

       /* for (size_t i = 1; i < hiddenY; i += 100)
        {
            printf("0 %i %f\n", i, tmpHiddenOutSignals1[i]);
        }*/

        for (size_t i = 1; i < hiddenX; ++i)
        {
            kernel_net_calculateLayer <<< numBlocks, blockSize >>> (weights, tmpHiddenOutSignals1, tmpHiddenOutSignals2, hiddenY, hiddenY, actPtr);
            weights += hiddenY * hiddenY;
            float* tmp = tmpHiddenOutSignals1;
            tmpHiddenOutSignals1 = tmpHiddenOutSignals2;
            tmpHiddenOutSignals2 = tmp;
            cudaDeviceSynchronize();
            /*for (size_t j = 0; j < hiddenY; j += 100)
            {
                float ttt = tmpHiddenOutSignals2[j];
                printf("%i %i %f\n", i,j, ttt);
            }*/
            //printf("v2 %f\n", v2);
            
        }

        numBlocks = (outputCount - 1) / blockSize + 1;
        kernel_net_calculateLayer <<< numBlocks, blockSize >>> (weights, tmpHiddenOutSignals1, outpuSignals, outputCount, hiddenY, actPtr);
        cudaDeviceSynchronize();
       /* for (size_t i = 0; i < outputCount; ++i)
        {
            printf("out %i %f\n", i, outpuSignals[i]);
        }*/
        delete[] tmpHiddenOutSignals1;
        delete[] tmpHiddenOutSignals2;

    }

    __global__ 
        void kernel_convertLayerWeightToGPUWeight(float* d_list, size_t signalCount, size_t neuronCount)
    {
        

        /*
        1) Kreisläufe finden:
            index 0 überspringen
            index 1 index speichern, kreislauf durchrechnen bis zum anfang, index in Liste aufnehmen
            index++ -> wieder index speichern, kreislauf druchrechnen bis anfang. bei jedem element prüfen, 
                       ob der jeweilige index schon in der Liste aufgenommen ist, wenn ja ist der Kreislauf
                       schon notiert.

            index++ -> wiederholen

            --> Ergiebt eine Liste von startindexen für die kreisläufe.

            --> 
        
        */

        size_t weightCount = signalCount * neuronCount;
        size_t circuitStartIndex[2];
        circuitStartIndex[0] = 0;
        circuitStartIndex[1] = 0;
        size_t circuitIndex = 0;

        // Kreisläufe finden
        // Erstes und letztes Elemnt überspringen, diese bleiben konstant.
        for (size_t currentStartIndex = 1; currentStartIndex < weightCount-1; ++currentStartIndex)
        {
            size_t destinationIndex = 0;
            size_t srcIndex = currentStartIndex;
            bool circuitAlreadyExists = false;
            while (destinationIndex != currentStartIndex) 
            {
                kernel_convertLayerWeightToGPUWeight_getNewIndex(srcIndex, destinationIndex, signalCount, neuronCount);
                
                for (int i = 0; i < 2; ++i)
                {
                    if (circuitStartIndex[i] == destinationIndex)
                    {
                        circuitAlreadyExists = true;
                    }
                }
                srcIndex = destinationIndex;
            }      
            if (!circuitAlreadyExists)
            {
                circuitStartIndex[circuitIndex] = currentStartIndex;
                ++circuitIndex;
                if (circuitIndex >= 2)
                {
                    printf("ERROR: more than 2 circuits found\n");
                }
            }
        }

/*
        size_t inputIndex = ;
        size_t outputIndex = inputIndex / neuronCount + (inputIndex % neuronCount) * signalCount;
        result[inputIndex / h][inputIndex % h] = m[x][y];*/

        /*
        size_t weightCount = signalCount * neuronCount -1;
        float tmp;
        size_t destIndex = 0;
        size_t srcIndex =  1;

        size_t srcIndexX = 1;
        size_t srcIncexY = 0;
        size_t destIndexX = 0;
        size_t destIncexY = 0;
        for (size_t x = 0; x <signalCount; ++x)
        {
            for (size_t y = 0; y < neuronCount; ++y)
            {
                if (x + y == 0 || x + y == signalCount + neuronCount-2)
                    continue;
                
               // destIndexX = (srcIndexX * signalCount) % signalCount;
               // destIndexY = (srcIndexY *)




                tmp = d_list[destIndex];
                d_list[destIndex] = d_list[srcIndex];
                destIndex = (destIndex + signalCount) % weightCount;
                ++srcIndex;
            }
        }*/
    }

    __global__ 
        void kernel_transposeMatrix(float* d_list, size_t width, size_t maxIndex, size_t indexOffset)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x + indexOffset;
       // #define COALESCED
        //maxIndex = kernel_gaussSum(width)
      
        size_t x = kernel_invGaussSum(index);
        size_t y = index - kernel_gaussSum(x);

        if (index >= maxIndex || x==y)
        {
            //printf(" returning index: %i\tx: %i\ty: %i\n", index, x, y);
            return;
        }
        if (y != 0)
            return;
       // if (index == 1)
        //    printf("maxIndex: %i\n", maxIndex);
        //if(x==0)
        //printf(" i: %3i x: %3i\n", index, x);

        if (x > width || y > width)
            printf("ERROR: x>width || y>width\n");
        size_t elementIndex1 = y * width + x;
        size_t elementIndex2 = x * width + y;
       // if (index == 2098176)
       //     printf("index == 2098176, d_list[%i][%i] = %1.1f\n", x, y, d_list[elementIndex1]);

       // if (elementIndex1 == 1 || elementIndex2 == 32)
      //      printf("");


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
        d_list[elementIndex1] = d_list[elementIndex2];
        d_list[elementIndex2] = tmp;
#endif
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
        return (size_t)floor((sqrt(8 * (float)sum + 1) - 1) / 2);
    }
    __device__ 
        size_t kernel_invGaussSum(size_t sum)
    {
        return (size_t)floorf((sqrtf(8 * (float)sum + 1) - 1) / 2);
    }


    __device__ 
        void kernel_convertLayerWeightToGPUWeight_getNewIndex(size_t startIndex, size_t& endIndex, size_t signalCount, size_t neuronCount)
    {
        // Calculates the new Position of a element
        endIndex = startIndex / neuronCount + (startIndex % neuronCount) * signalCount;
    }

}

