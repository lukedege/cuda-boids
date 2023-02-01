#pragma once

#include <vector>
#include <chrono>
#include <random>

#include <glm/glm.hpp>

#include "CUDA/vector_math.h"

namespace utils
{
	namespace time
	{
		inline double seconds() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
	}

	namespace math
	{
		struct plane
		{
			float4 origin;
			float4 normal;
		};

		inline __host__ __device__ int floor(const int x, const int y)
		{
			return x / y;
		}

		inline __host__ __device__ int ceil(const int x, const int y)
		{
			return static_cast<int>(::ceil(static_cast<double>(x) / y));
		}

		inline __host__ __device__ float normalized_value_in_range(const float val, const float min, const float max)
		{
			float result[] = { 0, (val - min) / (max - min) };
			int is_valid = max > min;
			return result[is_valid];
		}

#pragma region float3
		inline __host__ __device__ float length2(const float3 a)
		{
			return dot(a, a);
		}

		inline __host__ __device__ float distance2(const float3 a, const float3 b)
		{
			float3 diff = a - b;
			return length2(diff);
		}

		inline __host__ __device__ bool operator==(const float3 a, const float3 b)
		{
			return a.x == b.x && a.y == b.y && a.z == b.z;
		}

		inline __host__ __device__ bool operator!=(const float3 a, const float3 b)
		{
			return !operator==(a, b);
		}

		inline __host__ __device__ float3 normalize_or_zero_div(const float3 vec)
		{
			float3 zero{ 0 };
			if (vec == zero)
				return zero;
			else
				return normalize(vec);
		}

		inline __host__ __device__ float3 normalize_or_zero(const float3 vec)
		{
			float3 zero{ 0 }, normalized = normalize(vec);
			float3 result[] = { zero, normalized };
			int is_valid = vec != zero;
			return result[is_valid]; // has to be 0 or 1, branchless
		}

		inline __host__ __device__ float distance_point_plane(const float3 point, const plane plane)
		{
			float4 point4{ point.x, point.y, point.z, 1 };
			return dot(plane.normal, point4 - plane.origin);
		}

		inline __host__ __device__ void print_f3(const float3 f3)
		{
			printf("%.2f %.2f %.2f \n", f3.x, f3.y, f3.z);
		}
#pragma endregion

#pragma region float4
		inline __host__ __device__ float length2(const float4 a)
		{
			return dot(a, a);
		}

		inline __host__ __device__ float distance2(const float4 a, const float4 b)
		{
			float4 diff = a - b;
			return length2(diff);
		}

		inline __host__ __device__ bool operator==(const float4 a, const float4 b)
		{
			return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
		}

		inline __host__ __device__ bool operator!=(const float4 a, const float4 b)
		{
			return !operator==(a, b);
		}

		inline __host__ __device__ float4 normalize_or_zero_div(const float4 vec)
		{
			float4 zero{ 0 };
			if (vec == zero)
				return zero;
			else
				return normalize(vec);
		}

		inline __host__ __device__ float4 normalize_or_zero(const float4 vec)
		{
			float4 zero{ 0 }, normalized = normalize(vec);
			float4 result[] = { zero, normalized };
			int is_valid = vec != zero;
			return result[is_valid]; // has to be 0 or 1, branchless
		}

		inline __host__ __device__ float distance_point_plane(const float4 point, const plane plane)
		{
			return dot(plane.normal, point - plane.origin);
		}

		inline __host__ __device__ void print_f4(const float4 f4)
		{
			printf("%.2f %.2f %.2f %.2f \n", f4.x, f4.y, f4.z, f4.w);
		}
#pragma endregion
	}	
}