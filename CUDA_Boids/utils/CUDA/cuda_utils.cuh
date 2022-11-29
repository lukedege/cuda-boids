#pragma once

#include <iostream>
#include <random>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas.h>
#include <curand.h>
#include <cufft.h>
#include <cusparse.h>
#include <device_launch_parameters.h>

#include "../utils.h"
#include "../CUDA/vector_math.h"

namespace utils::cuda
{
	namespace math
	{
		struct plane
		{
			float4 origin;
			float4 normal;
		};

		inline __host__ __device__ int floor(int x, int y)
		{
			return x / y;
		}

		inline __host__ __device__ int ceil(int x, int y) 
		{
			return static_cast<int>(::ceil(static_cast<double>(x) / y));
		}

		inline __host__ __device__ float length2(float4 a)
		{
			return dot(a, a);
		}

		inline __host__ __device__ float distance2(float4 a, float4 b)
		{
			float4 diff = a - b;
			return length2(diff);
		}

		inline __host__ __device__ bool operator==(float4 a, float4 b)
		{
			return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
		}

		inline __host__ __device__ bool operator!=(float4 a, float4 b)
		{
			return !operator==(a, b);
		}

		inline __host__ __device__ float4 normalize_or_zero_div(float4 vec)
		{
			float4 zero{ 0 };
			if (vec == zero)
				return zero;
			else
				return normalize(vec);
		}

		inline __host__ __device__ float4 normalize_or_zero(float4 vec)
		{
			float4 zero{ 0 }, normalized = normalize(vec);
			float4 result[] = { zero, normalized };
			int is_valid = vec != zero;
			return result[is_valid]; // has to be 0 or 1, branchless
		}

		inline __host__ __device__ float distance_point_plane(float4 point, plane plane)
		{
			return dot(plane.normal, point - plane.origin);
		}

		inline __host__ __device__ float distance_point_plane(float4 point, utils::math::plane p)
		{
			float4 origin = { p.origin.r, p.origin.g, p.origin.b, p.origin.a };
			float4 normal = { p.normal.r, p.normal.g, p.normal.b, p.normal.a };
			plane converted{ origin, normal };
			return distance_point_plane(point, converted);
		}

		inline __host__ __device__ void print_f4(const float4 f4)
		{
			printf("%.2f %.2f %.2f %.2f \n", f4.x, f4.y, f4.z, f4.w);
		}
	}

	namespace containers
	{
		template<typename T>
		struct vec2
		{
			T x, y;

			T& operator[](const size_t index) { if (index) return y; else return x; }
		};


		template<typename T>
		struct sized_array
		{
			T* data;
			size_t size;
		};

		inline void random_int_fill_cpu(std::vector<int>& arr, const int range_from, const int range_to)
		{
			std::random_device                  rand_dev;
			std::mt19937                        generator(rand_dev());
			std::uniform_int_distribution<int>  distr(range_from, range_to);

			for (size_t i = 0; i < arr.size(); i++)
			{
				arr[i] = distr(generator);
			}
		}

		template<typename T>
		inline void random_vec2_fill_cpu(std::vector<vec2<T>>& arr, const int range_from, const int range_to)
		{
			std::random_device                  rand_dev;
			std::mt19937                        generator(rand_dev());
			std::uniform_int_distribution<int>  distr(range_from, range_to);

			for (size_t i = 0; i < arr.size(); i++)
			{
				arr[i][0] = distr(generator);
				arr[i][1] = distr(generator);
			}
		}

		template<typename T>
		inline void print(T* arr, size_t size)
		{
			for (size_t i = 0; i < size; i++)
			{
				std::cout << arr[i] << " ";
			}
			std::cout << std::endl;
		}

		template<typename Container>
		inline void print(const Container& data)
		{
			for (auto element : data)
			{
				std::cout << element << " ";
			}
			std::cout << std::endl;
		}
	}

	namespace checks
	{
		inline void check_cuda_result(const cudaError_t cuda_call_result)
		{
			if (cuda_call_result != cudaSuccess)
			{
				fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
				fprintf(stderr, "code: %d, reason: %s\n", cuda_call_result,
					cudaGetErrorString(cuda_call_result));
			}
		}

		inline void check_cublas_result(const cublasStatus_t cublas_call_result)
		{
			if (cublas_call_result != CUBLAS_STATUS_SUCCESS)
			{
				fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", cublas_call_result, __FILE__,
					__LINE__);
				exit(1);
			}
		}

		inline void check_curand_result(const curandStatus_t curand_call_result)
		{
			if (curand_call_result != CURAND_STATUS_SUCCESS)
			{
				fprintf(stderr, "Got CURAND error %d at %s:%d\n", curand_call_result, __FILE__,
					__LINE__);
				exit(1);
			}
		}

		inline void check_cufft_result(const cufftResult cufft_call_result)
		{
			if (cufft_call_result != CUFFT_SUCCESS)
			{
				fprintf(stderr, "Got CUFFT error %d at %s:%d\n", cufft_call_result, __FILE__,
					__LINE__);
				exit(1);
			}
		}

		inline void check_cusparse_result(const cusparseStatus_t cusparse_call_result)
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
			check_cuda_result(cudaGetDeviceProperties(&deviceProp, device_index));
			std::cout << "Selected device " << device_index << " " << deviceProp.name;
			check_cuda_result(cudaSetDevice(device_index));
		}
	}
}

// Workaround to fix intellisense parsing of the syncthreads function and of the kernel angled brackets
// e.g. my_kernel_name CUDA_KERNEL(grid_dim, block_dim, shared_mem, stream)(param1, param2, ...);
#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
void __syncthreads() {};
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif