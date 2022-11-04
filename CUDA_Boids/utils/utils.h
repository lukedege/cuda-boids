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

	namespace containers
	{
		inline void random_vec2_fill_cpu(std::vector<glm::vec2>& arr, const int range_from, const int range_to)
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
	}
	
	namespace gl
	{
		inline void setup_ssbo(GLuint& ssbo, int alloc_size, int bind_index, void* data)
		{
			glGenBuffers(1, &ssbo);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);

			glBufferData(GL_SHADER_STORAGE_BUFFER, alloc_size, NULL, GL_DYNAMIC_DRAW); // allocate alloc_size bytes of memory
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind_index, ssbo);

			if (data != 0)
				glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, alloc_size, data);        // fill buffer object with data

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		}
	}
}