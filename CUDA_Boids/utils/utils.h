#pragma once

#include <vector>
#include <chrono>
#include <random>

#include <glm/glm.hpp>

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
			glm::vec4 origin;
			glm::vec4 normal;
		};

		inline glm::vec4 normalize(glm::vec4 vec)
		{
			glm::vec4 zero{ 0 };
			if (vec != zero)
				return glm::normalize(vec);
			else
				return zero;
		}

		inline float distance_point_plane(glm::vec4 point, plane plane)
		{
			return glm::dot(plane.normal, point - plane.origin);
		}
	}	

	namespace containers
	{
		inline void random_vec2_fill_cpu(std::vector<glm::vec2>& arr, const int range_from, const int range_to)
		{
			std::random_device                     rand_dev;
			std::mt19937                           generator(rand_dev());
			std::uniform_real_distribution<float>  distr(range_from, range_to);

			for (size_t i = 0; i < arr.size(); i++)
			{
				arr[i][0] = distr(generator);
				arr[i][1] = distr(generator);
			}
		}
		inline void random_vec3_fill_cpu(std::vector<glm::vec3>& arr, const int range_from, const int range_to)
		{
			std::random_device                     rand_dev;
			std::mt19937                           generator(rand_dev());
			std::uniform_real_distribution<float>  distr(range_from, range_to);

			for (size_t i = 0; i < arr.size(); i++)
			{
				arr[i][0] = distr(generator);
				arr[i][1] = distr(generator);
				arr[i][2] = distr(generator);
			}
		}
		inline void random_vec4_fill_cpu(std::vector<glm::vec4>& arr, const int range_from, const int range_to)
		{
			std::random_device                     rand_dev;
			std::mt19937                           generator(rand_dev());
			std::uniform_real_distribution<float>  distr(range_from, range_to);

			for (size_t i = 0; i < arr.size(); i++)
			{
				arr[i][0] = distr(generator);
				arr[i][1] = distr(generator);
				arr[i][2] = distr(generator);
				arr[i][3] = 0;
			}
		}
	}
}