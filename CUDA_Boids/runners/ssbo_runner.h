#pragma once
#include "boid_runner.h"

namespace utils::runners
{
	class ssbo_runner : public boid_runner
	{
	protected:
		ssbo_runner(simulation_parameters params = {}) :
			boid_runner    { {"shaders/ssbo.vert", "shaders/basic.frag"}, params},
			triangle_mesh  { setup_mesh() },
			ssbo_positions { 0 },
			ssbo_velocities{ 0 } {}

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

		utils::graphics::opengl::Mesh triangle_mesh;

		GLuint ssbo_positions; 
		GLuint ssbo_velocities;
	};
}