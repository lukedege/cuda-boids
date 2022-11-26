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
		ssbo_runner(){}
		ssbo_runner(simulation_parameters params) : boid_runner{params}{}

		inline utils::graphics::opengl::Mesh setup_mesh()
		{
			std::vector<utils::graphics::opengl::Vertex> vertices
			{
				{{  0.0f,  0.5f, 0.0f,}},
				{{  0.3f, -0.3f, 0.0f }},
				{{ -0.3f, -0.3f, 0.0f }}
			};
			std::vector<GLuint>  indices {0, 1, 2};
			return utils::graphics::opengl::Mesh(vertices, indices);
		}

	};
}