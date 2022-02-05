#include "../inc/net_kernel.cuh"

namespace NeuronalNet 
{
    __host__ 
        void GPU_CUDA_memcpyTest()
    {
        {
            size_t count = 8;
            float* h_original = new float[count];
            float* h_check = new float[count];

            float* d_original;
            float* d_check;

            cudaMalloc(&d_original, count * sizeof(float));
            cudaMalloc(&d_check, count * sizeof(float));


            for (size_t i = 0; i < count; ++i)
            {
                h_original[i] = 1.11 * (i + 1);
            }

            std::cout << "origninal: \n";
            for (size_t i = 0; i < count; ++i)
            {
                std::cout << "i = " << i << "\t" << h_original[i] << "\t" << h_check[i] << "\n";
            }

            cudaMemcpy(d_original, h_original, count * sizeof(float), cudaMemcpyHostToDevice);
            std::cout << "Kernel call: \n";
            kernel_memcpyTest1 << <1, 1 >> > (d_original, d_check, count);
            cudaDeviceSynchronize();
            cudaMemcpy(h_check, d_check, count * sizeof(float), cudaMemcpyDeviceToHost);

            std::cout << "check: \n";
            for (size_t i = 0; i < count; ++i)
            {
                std::cout << "i = " << i << "\t" << h_original[i] << "\t" << h_check[i] << "\n";
            }

            cudaFree(d_original);
            cudaFree(d_check);
            delete[] h_original;
            delete[] h_check;
        }
        {
            size_t count = 8;
            float* h_original = new float[count];
            float* h_check = new float[count];

            float* d_original;
            float* d_check;

            cudaMalloc(&d_original, count * sizeof(float));
            cudaMalloc(&d_check, count * sizeof(float));


            for (size_t i = 0; i < count; ++i)
            {
                h_original[i] = 1.11 * (i + 1);
            }

            std::cout << "origninal: \n";
            for (size_t i = 0; i < count; ++i)
            {
                std::cout << "i = " << i << "\t" << h_original[i] << "\t" << h_check[i] << "\n";
            }

            cudaMemcpy(d_original, h_original, count * sizeof(float), cudaMemcpyHostToDevice);
            std::cout << "Kernel call: \n";
            kernel_memcpyTest2 << <1, 1 >> > (d_original, d_check, count);
            cudaDeviceSynchronize();
            cudaMemcpy(h_check, d_check, count * sizeof(float), cudaMemcpyDeviceToHost);

            std::cout << "check: \n";
            for (size_t i = 0; i < count; ++i)
            {
                std::cout << "i = " << i << "\t" << h_original[i] << "\t" << h_check[i] << "\n";
            }

            cudaFree(d_original);
            cudaFree(d_check);
            delete[] h_original;
            delete[] h_check;
        }

    }
    struct Storage
    {
        float a1;
     //   float a2;
      //  float a3;
      //  float a4;
        //float a5;
        //float a6;
       // float a7;
      //  float a8;
    };
    __global__ 
        void kernel_memcpyTest1(float* ptrA, float* ptrB, size_t count)
    {
        Storage* dest = (Storage*)ptrB;
        *dest = *(Storage*)ptrA;
    }
    __global__
        void kernel_memcpyTest2(float* ptrA, float* ptrB, size_t count)
    {
       /* float a0;
        asm volatile ("mov.f32 %0, %1;"
            : "=f"(a0)
            : "f"(*ptrA));


        asm volatile ("mov.f32 %0, %1;"
            : "=f"(a0)
            : "f"(a0));

        double x;
        double x1 = 11;

        asm volatile ("mov.f64 %0, %1;"
            : "=d"(x)
            : "d"(x1));

        //printf("size: %i\n", sizeof(float));
        //printf("size: %i\n", sizeof(double));

        float a1 = ptrA[1];
        float a2 = ptrA[2];
        float a3 = ptrA[3];
        float a4 = ptrA[4];
        float a5 = ptrA[5];
        float a6 = ptrA[6];
        float a7 = ptrA[7];*/

        ptrB[0] = ptrA[0];
        ptrB[1] = ptrA[1];
        ptrB[2] = ptrA[2];
        ptrB[3] = ptrA[3];
        ptrB[4] = ptrA[4];
        ptrB[5] = ptrA[5];
        ptrB[6] = ptrA[6];
        ptrB[7] = ptrA[7];
    }


    __host__ 
        void GPU_CUDA_getSpecs()
    {
        cudaDeviceProp h_deviceProp;
        cudaGetDeviceProperties(&h_deviceProp, 0);
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

      
        size_t storageCount = 0;
        Storage storage;
        Storage *ramStorage;

        float res = 0;
        weights += index * inputSignalCount;

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
                
                
                float weight = weights[i];
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
    /*__global__
        void kernel_scaleRandomWeight(float min, float max, float* d_list, size_t elements)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= elements)
            return;

        size_t r = (size_t)d_list[index];
        d_list[index] = min + (r % (size_t(max) - size_t(min)));
        //if(index == 0)
            //curandGenerateUniform(*gen, d_list, elements);
        //cudaDeviceSynchronize();
    }*/
}


