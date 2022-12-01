#pragma once
#include "vao_runner.h"

#include "../utils/CUDA/cudaGLmanager.h"

namespace utils::runners
{
	class gpu_vao : public vao_runner
	{
	public:
		gpu_vao(simulation_parameters params);

		void calculate(const float delta_time);

		void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix);

	private:
		size_t amount;
		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> velocities;

		size_t block_size;
		size_t grid_size;

		utils::cuda::gl_manager cuda_gl_manager;
		GLuint  vao;
		float3* vbo_positions_dptr;
		float3* vbo_velocities_dptr;
	};
}