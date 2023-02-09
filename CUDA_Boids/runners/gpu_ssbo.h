#pragma once
#include "ssbo_runner.h"

#include "../utils/CUDA/cudaGLmanager.h"
#include "gpu_boid_behaviours.h"

namespace utils::runners
{
	class gpu_ssbo : public ssbo_runner
	{
	public:
		gpu_ssbo(simulation_parameters params);

		void calculate(const float delta_time);

		void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix);

		~gpu_ssbo();

		simulation_parameters get_simulation_parameters();
		void set_dynamic_simulation_parameters(simulation_parameters::dynamic_parameters new_dyn_params);

	private:
		void naive_calculation(const float delta_time);
		void uniform_grid_calculation(const float delta_time);
		void coherent_grid_calculation(const float delta_time);

		size_t amount;

		float4* ssbo_positions_dptr;
		float4* ssbo_velocities_dptr;

		size_t block_size;
		size_t grid_size;
		utils::cuda::gl_manager cuda_gl_manager;

		// Fields for modular method (but unviable, kept for reference)
		//cudaStream_t ali_stream, coh_stream, sep_stream, wsp_stream;
		//float4* alignments_dptr;
		//float4* cohesions_dptr;
		//float4* separations_dptr;
		//float4* wall_separations_dptr;

		// Grid-related fields
		float grid_resolution;
		//behaviours::boid_cell_index* boid_cell_indices_dptr; // aka bci
		//behaviours::idx_range* cell_idx_range_dptr; // aka cir, AoS pattern = bad (kept for reference)
		int* bci_boid_indices_dptr;
		int* bci_cell_indices_dptr; // split into two separate arrays to enforce SoA memory access pattern
		
		int* cell_idx_range_start_dptr; // same reason as above
		int* cell_idx_range_end_dptr;
		

		cudaStream_t bci_stream, cir_stream, cir_stream2;

		// Swap-auxiliary arrays
		float4* sorted_positions_dptr;
		float4* sorted_velocities_dptr;

		cudaStream_t pos_stream, vel_stream;
	};
}