#pragma once
#include "boid_runner.h"

#include "../utils/CUDA/cudaGLmanager.h"
#include "../utils/shader.h"
#include "../utils/mesh.h"
#include "../utils/flock.h"

namespace utils::runners
{
	class gpu_vel_based : public boid_runner
	{
	public:
		gpu_vel_based(utils::graphics::opengl::Mesh& mesh, const size_t amount);

		void calculate(const float delta_time);

		void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix);

	private:
		utils::graphics::opengl::Shader shader;

		size_t amount;
		utils::graphics::opengl::Flock triangles;

		size_t block_size;
		size_t grid_size;

		utils::cuda::gl_manager cuda_gl_manager;
		float2* ssbo_positions_dptr;
		float2* ssbo_velocities_dptr;
	};
}