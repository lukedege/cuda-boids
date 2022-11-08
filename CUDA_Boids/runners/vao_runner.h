#pragma once
#include "boid_runner.h"

#include "../utils/shader.h"
#include "../utils/mesh.h"
#include "../utils/flock.h"

namespace utils::runners
{
	class vao_runner : public boid_runner
	{
	protected:

		inline void setup_vbo(GLuint& vbo, size_t element_size, size_t element_amount, int bind_index, void* data)
		{
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);

			size_t alloc_size = element_size * element_amount;
			glBufferData(GL_ARRAY_BUFFER, alloc_size, NULL, GL_DYNAMIC_DRAW);

			if (data != 0)
				glBufferSubData(GL_ARRAY_BUFFER, 0, alloc_size, data);

			glEnableVertexAttribArray(bind_index);
			glVertexAttribPointer(bind_index, 3, GL_FLOAT, GL_FALSE, element_size, (GLvoid*)0);

			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
	};
}