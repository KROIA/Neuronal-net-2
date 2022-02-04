#include "../inc/net_kernel.cuh"

namespace NeuronalNet 
{
    __host__ 
        void GPU_CUDA_calculateNet(float* weights, float* signals, float* outpuSignals,
                               size_t inputCount, size_t hiddenX, size_t hiddenY, size_t outputCount, Activation activation)
    {
        kernel_calculateNet << <1, 1 >> > (weights, signals, outpuSignals,
                                           inputCount, hiddenX, hiddenY, outputCount, activation);
        cudaDeviceSynchronize();
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

        const size_t tileSize = 2048*2;
        __shared__ float sharedSignals[tileSize];
        //printf("B %i   Bd %i  T %i  Index: %i\n", blockIdx.x, blockDim.x, threadIdx.x, index);
        float res = 0;
        size_t signalsBegin = 0;
        size_t signalsEnd   = 0;
        size_t tiles = inputSignalCount / tileSize + 1;
        size_t loadCount = tileSize / neuronCount + 1;
        //float delta;
        for (size_t tile = 0; tile <tiles; ++tile)
        {
            signalsBegin = signalsEnd;
            if (tile == tiles - 1)
                signalsEnd = inputSignalCount;
            else
                signalsEnd = (tile + 1) * tileSize;

            /*if (index < signalsEnd - signalsBegin)
            {
                sharedSignals[index] = inputSignals[i];
            }*/
            
            for (size_t i = 0; i < loadCount; ++i)
            {
                size_t signalIndex = i + loadCount * threadIdx.x;

                if (signalsEnd > (signalIndex + signalsBegin))
                {
                     sharedSignals[signalIndex] = inputSignals[signalIndex+signalsBegin];
                }
            }
            

            /*if (threadIdx.x == 0)
            {
                
                for (size_t i = signalsBegin; i < signalsEnd; ++i)
                {
                    sharedSignals[i- signalsBegin] = inputSignals[i];
                }
                
            }*/
            __syncthreads();


            
            for (size_t i = signalsBegin; i < signalsEnd; ++i)
            {
                //float weight = weights[index * inputSignalCount + i];
                //float signal = sharedSignals[i - signalsBegin];
                //delta = weight * signal;
                //if (delta < 0.00005 && delta > -0.00005)
                //    delta = 0;
                //else
                res += weights[index * inputSignalCount + i] * sharedSignals[i - signalsBegin];
               /* if (weights[index * inputSignalCount + i] != 0.01)
                {
                    printf("A: %f\n", weights[index * inputSignalCount + i]);
                }*/
                /*if ((delta - 1) > 0.0001 || (delta - 1) < -0.0001)
                {
                    delta = delta;
                }
                else if (delta == 0)
                {
                    delta = delta;
                }*/
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
        kernel_net_calculateLayer <<< numBlocks, blockSize >>> (weights, signals, tmpHiddenOutSignals1, hiddenY, inputCount, actPtr);
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


