#pragma once
#include "boid_runner.h"

#include "../utils/shader.h"
#include "../utils/mesh.h"
#include "../utils/flock.h"

namespace utils::runners
{
	class ssbo_runner : public boid_runner
	{
	protected:
		inline utils::graphics::opengl::Mesh setup_mesh()
		{
			std::vector<utils::graphics::opengl::Vertex> vertices
			{
				{{  0.0f,  0.5f, 0.0f,}},
				{{  0.5f, -0.5f, 0.0f }},
				{{ -0.5f, -0.5f, 0.0f }}
			};
			std::vector<GLuint>  indices {0, 1, 2};
			return utils::graphics::opengl::Mesh(vertices, indices);
		}

		inline void setup_ssbo(GLuint& ssbo, size_t element_size, size_t element_amount, int bind_index, void* data)
		{
			glGenBuffers(1, &ssbo);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);

			size_t alloc_size = element_size * element_amount;
			glBufferData(GL_SHADER_STORAGE_BUFFER, alloc_size, NULL, GL_DYNAMIC_DRAW); // allocate alloc_size bytes of memory
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind_index, ssbo);

			if (data != 0)
				glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, alloc_size, data);        // fill buffer object with data

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		}
	};
}