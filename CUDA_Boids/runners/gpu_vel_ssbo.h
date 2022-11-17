#pragma once
#include "ssbo_runner.h"

#include "../utils/CUDA/cudaGLmanager.h"
#include "../utils/shader.h"
#include "../utils/mesh.h"
#include "../utils/flock.h"

namespace utils::runners
{
	class gpu_vel_ssbo : public ssbo_runner
	{
	public:
		gpu_vel_ssbo(const size_t amount);

		void calculate(const float delta_time);

		void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix);

	private:
		utils::graphics::opengl::Shader shader;

		size_t amount;
		utils::graphics::opengl::Mesh triangle_mesh;
		std::vector<glm::vec4> positions;
		std::vector<glm::vec4> velocities;

		size_t block_size;
		size_t grid_size;

		utils::cuda::gl_manager cuda_gl_manager;
		float4* ssbo_positions_dptr;
		float4* ssbo_velocities_dptr;
	};
}