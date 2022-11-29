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
		gpu_vel_ssbo(simulation_parameters params);

		void calculate(const float delta_time);

		void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix);

		~gpu_vel_ssbo();

		simulation_parameters get_simulation_parameters();
		void set_simulation_parameters(simulation_parameters new_params);

	private:
		void naive_calculation(const float delta_time);

		utils::graphics::opengl::Shader shader;

		size_t amount;
		utils::graphics::opengl::Mesh triangle_mesh;
		std::vector<glm::vec4> positions;
		std::vector<glm::vec4> velocities;

		size_t block_size;
		size_t grid_size;

		utils::cuda::gl_manager cuda_gl_manager;
		simulation_parameters* sim_params_dptr;
		utils::math::plane* sim_volume_dptr;

		float4* ssbo_positions_dptr;
		float4* ssbo_velocities_dptr;

		float4* alignments_dptr;
		float4* cohesions_dptr;
		float4* separations_dptr;
		float4* wall_separations_dptr;
	};
}