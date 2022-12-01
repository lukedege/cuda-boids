#pragma once

#include <iostream>
#include <random>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <cusparse.h>
#include <device_launch_parameters.h>

#include "../utils.h"

// Workaround to fix intellisense parsing of the syncthreads function and of the kernel angled brackets
// e.g. my_kernel_name CUDA_KERNEL(grid_dim, block_dim, shared_mem, stream)(param1, param2, ...);
#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
	void __syncthreads() {};
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

namespace utils::cuda
{
	namespace checks
	{
		inline __host__ __device__ void cuda(const cudaError_t cuda_call_result)
		{
			if (cuda_call_result != cudaSuccess)
			{
				fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
				fprintf(stderr, "code: %d, reason: %s\n", cuda_call_result,
					cudaGetErrorString(cuda_call_result));
			}
		}

		inline __host__ __device__ void cublas(const cublasStatus_t cublas_call_result)
		{
			if (cublas_call_result != CUBLAS_STATUS_SUCCESS)
			{
				fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", cublas_call_result, __FILE__,
					__LINE__);
				exit(1);
			}
		}

		inline __host__ __device__ void curand(const curandStatus_t curand_call_result)
		{
			if (curand_call_result != CURAND_STATUS_SUCCESS)
			{
				fprintf(stderr, "Got CURAND error %d at %s:%d\n", curand_call_result, __FILE__,
					__LINE__);
				exit(1);
			}
		}

		inline __host__ __device__ void cufft(const cufftResult cufft_call_result)
		{
			if (cufft_call_result != CUFFT_SUCCESS)
			{
				fprintf(stderr, "Got CUFFT error %d at %s:%d\n", cufft_call_result, __FILE__,
					__LINE__);
				exit(1);
			}
		}

		inline __host__ __device__ void cusparse(const cusparseStatus_t cusparse_call_result)
		{
			if (cusparse_call_result != CUSPARSE_STATUS_SUCCESS)
			{
				fprintf(stderr, "Got error %d at %s:%d\n", cusparse_call_result, __FILE__, __LINE__);
				cudaError_t cuda_err = cudaGetLastError();
				if (cuda_err != cudaSuccess)
				{
					fprintf(stderr, "  CUDA error \"%s\" also detected\n",
						cudaGetErrorString(cuda_err));
				}
				exit(1);
			}
		}

		inline void select_device(int device_index = 0)
		{
			// set up device
			cudaDeviceProp deviceProp;
			cuda(cudaGetDeviceProperties(&deviceProp, device_index));
			std::cout << "Selected device " << device_index << " " << deviceProp.name;
			cuda(cudaSetDevice(device_index));
		}
	}

	namespace containers
	{
		namespace 
		{
			inline __global__ void setup_states(curandState* state)
			{
				int id = threadIdx.x + blockIdx.x * blockDim.x;
				/* Each thread gets same seed, a different sequence
				   number, no offset */
				curand_init(1234, id, 0, &state[id]);
			}

			inline __global__ void generate_float(curandState* global_state, float* arr, const size_t size, const int min, const int max)
			{
				int id = threadIdx.x + blockIdx.x * blockDim.x;
				if (id >= size) return;

				float x;
				/* Copy state to local memory for efficiency */
				curandState local_state = global_state[id];
				/* Generate pseudo-random uniforms */
				x = curand_uniform(&local_state) * (max - min) + min;
				/* Copy state back to global memory */
				global_state[id] = local_state;
				/* Store results */
				arr[id] = x;
			}

			inline __global__ void generate_float4(curandState* global_state, float4* arr, const size_t size, const int min, const int max)
			{
				int id = threadIdx.x + blockIdx.x * blockDim.x;
				if (id >= size) return;

				float4 res;
				/* Copy state to local memory for efficiency */
				curandState local_state = global_state[id];
				/* Generate pseudo-random uniforms */
				res.x = curand_uniform(&local_state) * (max - min) + min;
				res.y = curand_uniform(&local_state) * (max - min) + min;
				res.z = curand_uniform(&local_state) * (max - min) + min;
				res.w = 0;
				/* Copy state back to global memory */
				global_state[id] = local_state;
				/* Store results */
				arr[id] = res;
			}
		}

		// takes a device pointer to the array (dptr for short)
		inline __host__ void random_vec4_fill_dptr(float4* arr_dptr, const size_t size, const int min, const int max)
		{
			size_t block_size{ 32 };
			size_t grid_size { static_cast<size_t>(utils::math::ceil(size, block_size)) };

			curandState* states_dptr;
			checks::cuda(cudaMalloc((void**)&states_dptr, size * sizeof(curandState)));
			setup_states CUDA_KERNEL(grid_size, block_size)(states_dptr);
			cudaDeviceSynchronize(); // TODO use events maybe
			generate_float4 CUDA_KERNEL(grid_size, block_size)(states_dptr, arr_dptr, size, min, max);
			cudaDeviceSynchronize();
			checks::cuda(cudaFree(states_dptr));
		}

		// takes a host pointer to the array (hptr for short)
		inline __host__ void random_vec4_fill_hptr(float4* arr_hptr, const size_t size, const int min, const int max)
		{
			float4* arr_dptr;
			checks::cuda(cudaMalloc((void**)&arr_dptr, size * sizeof(float4)));
			random_vec4_fill_dptr(arr_dptr, size, min, max);
			checks::cuda(cudaMemcpy(arr_hptr, arr_dptr, size * sizeof(float4), cudaMemcpyDeviceToHost));
			checks::cuda(cudaFree(arr_dptr));
		}
	}
}