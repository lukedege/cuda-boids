#include "cpu_ssbo.h"

// std libraries
#include <vector>
#include <math.h>

#include <glad.h>
#include <glm/glm.hpp>

// utils libraries
#include "../utils/utils.h"
#include "../utils/CUDA/cuda_utils.h"
#include "cpu_boid_behaviours.h"

namespace utils::runners
{
	cpu_ssbo::cpu_ssbo(simulation_parameters params) :
		ssbo_runner{ {"shaders/ssbo.vert", "shaders/basic.frag"}, params },
		amount{ params.static_params.boid_amount },
		positions { std::vector<float4>(amount) },
		velocities{ std::vector<float4>(amount) }
	{
		float spawn_range = sim_params.static_params.cube_size * 0.5f - 0.001f;
		utils::cuda::containers::random_vec4_fill_hptr(positions.data(), amount, -spawn_range, spawn_range);
		utils::cuda::containers::random_vec4_fill_hptr(velocities.data(), amount, -1, 1);

		setup_buffer_object(ssbo_positions , GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 0, positions.data());
		setup_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 1, velocities.data());
	}

	void cpu_ssbo::naive_calculation(const float delta_time)
	{
		namespace n_bhvr = behaviours::naive::cpu;
		float4 accel_blend;
		float chs = sim_params.static_params.cube_size / 2;
		float boid_fov = sim_params.dynamic_params.boid_fov;
		for (size_t i = 0; i < amount; i++)
		{
			accel_blend = sim_params.dynamic_params.alignment_coeff       * n_bhvr::alignment(i, positions.data(), velocities.data(), amount, boid_fov)
						+ sim_params.dynamic_params.cohesion_coeff        * n_bhvr::cohesion(i, positions.data(), amount, boid_fov)
						+ sim_params.dynamic_params.separation_coeff      * n_bhvr::separation(i, positions.data(), amount, boid_fov)
						+ sim_params.dynamic_params.wall_separation_coeff * n_bhvr::wall_separation(i, positions.data(), sim_volume.data(), amount);

			//velocities[i] = normalize(velocities[i]) + normalize(accel_blend) * delta_time; //v = u + at
			velocities[i] = utils::math::normalize_or_zero(velocities[i] + accel_blend * delta_time); //v = u + at
			positions[i] += velocities[i] * sim_params.dynamic_params.boid_speed * delta_time; //s = vt
			positions[i] = clamp(positions[i], { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
		}

		update_buffer_object(ssbo_positions , GL_SHADER_STORAGE_BUFFER, 0, sizeof(float4), amount, positions.data());
		update_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, 0, sizeof(float4), amount, velocities.data());
	}

	namespace
	{
		std::vector<behaviours::grid::boid_cell_index> assign_grid_indices(const float4* boid_positions, const size_t boid_amount, const float grid_extent, const float grid_resolution)
		{
			std::vector<behaviours::grid::boid_cell_index> boid_cell_indices(boid_amount);

			int cell_amount = grid_resolution * grid_resolution * grid_resolution;
			float cube_half_size = grid_extent / 2;

			int x, y, z, linear_index;
			for (size_t i = 0; i < boid_amount; i++)
			{
				x = utils::math::normalized_value_in_range(boid_positions[i].x, -cube_half_size, cube_half_size) * grid_resolution;
				y = utils::math::normalized_value_in_range(boid_positions[i].y, -cube_half_size, cube_half_size) * grid_resolution;
				z = utils::math::normalized_value_in_range(boid_positions[i].z, -cube_half_size, cube_half_size) * grid_resolution;
				linear_index = x * grid_resolution * grid_resolution + y * grid_resolution + z;
				boid_cell_indices[i].cell_id = std::clamp(linear_index, 0, cell_amount - 1);
				boid_cell_indices[i].boid_id = i;
			}

			return boid_cell_indices;
		}

		std::vector<behaviours::grid::idx_range> find_cell_boid_range(const behaviours::grid::boid_cell_index* boid_cell_indices, const size_t boid_amount, const float grid_resolution)
		{
			int cell_amount = grid_resolution * grid_resolution * grid_resolution;

			std::vector<behaviours::grid::idx_range> cell_idx_range(cell_amount);

			int current_cell = 0, start = 0, read_cell = 0;
			size_t i;
			for (i = 0; i < boid_amount; i++)
			{
				read_cell = boid_cell_indices[i].cell_id;
				if (read_cell != current_cell)
				{
					cell_idx_range[current_cell].start = start;
					cell_idx_range[current_cell].end = i;
					current_cell = read_cell;
					start = i;
				}
			}
			// for last cell
			if (current_cell < cell_idx_range.size())
			{
				cell_idx_range[current_cell].start = start;
				cell_idx_range[current_cell].end = i;
			}

			return cell_idx_range;
		}
	}

	void cpu_ssbo::uniform_grid_calculation(const float delta_time)
	{
		namespace grid_bhvr = behaviours::grid;

		float boid_fov = sim_params.dynamic_params.boid_fov;
		float cell_size = 2 * boid_fov; // we base our grid size based on the boid fov 
		float grid_resolution = sim_params.static_params.cube_size / cell_size;
		
		// 2) create a "grid" and localize each boid inside of it with a linear index
		std::vector<grid_bhvr::boid_cell_index> boid_cell_indices{ assign_grid_indices(positions.data(), amount, sim_params.static_params.cube_size, grid_resolution) };
		
		// 3) sort the boids according to their linear index 
		auto order_by_cell_id = [](const grid_bhvr::boid_cell_index& a, const grid_bhvr::boid_cell_index& b) -> bool { return a.cell_id < b.cell_id; };
		std::sort(boid_cell_indices.begin(), boid_cell_indices.end(), order_by_cell_id);
		
		// 4) calculate "start" and "end" indices for each cell (to find the range of adjacent, same-cell boids)
		std::vector<grid_bhvr::idx_range> cell_idx_range{ find_cell_boid_range(boid_cell_indices.data(), amount, grid_resolution) };

		// 5) calcola le velocit� usando solo quella cella come neighborhood
		float4 accel_blend;
		float chs = sim_params.static_params.cube_size / 2 - 0.0001f;
		for (size_t i = 0; i < amount; i++)
		{
			grid_bhvr::boid_cell_index current = boid_cell_indices[i];
			accel_blend = sim_params.dynamic_params.alignment_coeff       * grid_bhvr::uniform::cpu::alignment      (current.boid_id, positions.data(), velocities.data(), boid_cell_indices.data(), cell_idx_range[current.cell_id].start, cell_idx_range[current.cell_id].end, boid_fov)
						+ sim_params.dynamic_params.cohesion_coeff        * grid_bhvr::uniform::cpu::cohesion       (current.boid_id, positions.data(), boid_cell_indices.data(), cell_idx_range[current.cell_id].start, cell_idx_range[current.cell_id].end, boid_fov)
						+ sim_params.dynamic_params.separation_coeff      * grid_bhvr::uniform::cpu::separation     (current.boid_id, positions.data(), boid_cell_indices.data(), cell_idx_range[current.cell_id].start, cell_idx_range[current.cell_id].end, boid_fov)
						+ sim_params.dynamic_params.wall_separation_coeff * grid_bhvr::uniform::cpu::wall_separation(current.boid_id, positions.data(), sim_volume.data());

			velocities[current.boid_id] = utils::math::normalize_or_zero(velocities[current.boid_id] + accel_blend * delta_time); //v = u + at
			positions [current.boid_id] += velocities[current.boid_id] * sim_params.dynamic_params.boid_speed * delta_time; //s = vt
			positions [current.boid_id] = clamp(positions[current.boid_id], { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
		}

		update_buffer_object(ssbo_positions, GL_SHADER_STORAGE_BUFFER, 0, sizeof(float4), amount, positions.data());
		update_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, 0, sizeof(float4), amount, velocities.data());
	}

	void cpu_ssbo::coherent_grid_calculation(const float delta_time)
	{
		namespace grid_bhvr = behaviours::grid;

		float boid_fov = sim_params.dynamic_params.boid_fov;
		float cell_size = 2 * boid_fov; // we base our grid size based on the boid fov 
		float grid_resolution = sim_params.static_params.cube_size / cell_size;
		
		// 2) create a "grid" and localize each boid inside of it with a linear index
		std::vector<grid_bhvr::boid_cell_index> boid_cell_indices{ assign_grid_indices(positions.data(), amount, sim_params.static_params.cube_size, grid_resolution) };
		
		// 3) ordina l'array di boid secondo la loro posizione di griglia(scattered=ordina solo il boid index, coherent = ordina pure velocities e positions)
		auto order_by_cell_id = [](const grid_bhvr::boid_cell_index& a, const grid_bhvr::boid_cell_index& b) -> bool { return a.cell_id < b.cell_id; };
		std::sort(boid_cell_indices.begin(), boid_cell_indices.end(), order_by_cell_id);
		
		std::vector<float4> new_vel(amount), new_pos(amount);
		std::vector<int> cell_ids(amount);
		for (size_t i = 0; i < amount; i++)
		{
			new_vel[i]  = velocities[boid_cell_indices[i].boid_id];
			new_pos[i]  = positions[boid_cell_indices[i].boid_id];
			cell_ids[i] = boid_cell_indices[i].cell_id;
		}

		velocities = new_vel;
		positions = new_pos;
		
		// 4) calcola "start" e "end" di ogni cella della griglia(ovvero il range di indici uguali dei boid adiacenti)
		std::vector<grid_bhvr::idx_range> cell_idx_range{ find_cell_boid_range(boid_cell_indices.data(), amount, grid_resolution) };

		// 5) calcola le velocit� usando solo quella cella come neighborhood
		float4 accel_blend;
		float chs = sim_params.static_params.cube_size / 2 - 0.0001f; 
		for (size_t i = 0; i < amount; i++)
		{
			accel_blend = sim_params.dynamic_params.alignment_coeff       * grid_bhvr::coherent::cpu::alignment      (i, positions.data(), velocities.data(), cell_ids.data(), cell_idx_range.data(), boid_fov)
						+ sim_params.dynamic_params.cohesion_coeff        * grid_bhvr::coherent::cpu::cohesion       (i, positions.data(), cell_ids.data(), cell_idx_range.data(), boid_fov)
						+ sim_params.dynamic_params.separation_coeff      * grid_bhvr::coherent::cpu::separation     (i, positions.data(), cell_ids.data(), cell_idx_range.data(), boid_fov)
						+ sim_params.dynamic_params.wall_separation_coeff * grid_bhvr::coherent::cpu::wall_separation(i, positions.data(), sim_volume.data());

			velocities[i] = utils::math::normalize_or_zero(velocities[i] + accel_blend * delta_time); //v = u + at
			positions [i] += velocities[i] * sim_params.dynamic_params.boid_speed * delta_time; //s = vt
			positions [i] = clamp(positions[i], { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
		}

		update_buffer_object(ssbo_positions, GL_SHADER_STORAGE_BUFFER, 0, sizeof(float4), amount, positions.data());
		update_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, 0, sizeof(float4), amount, velocities.data());
	}

	void cpu_ssbo::calculate(const float delta_time)
	{
		//naive_calculation(delta_time);
		uniform_grid_calculation(delta_time);
		//coherent_grid_calculation(delta_time);
	}

	void cpu_ssbo::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		// Update references to view and projection matrices
		update_buffer_object(ubo_matrices, GL_UNIFORM_BUFFER, 0                , sizeof(glm::mat4), 1, (void*) glm::value_ptr(view_matrix)      );
		update_buffer_object(ubo_matrices, GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), 1, (void*) glm::value_ptr(projection_matrix));

		// Setup and draw debug info (simulation volume, ...)
		debug_shader.use();
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		cube_mesh.draw(GL_LINES);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		// Setup and draw boids
		boid_shader.use();
		triangle_mesh.draw_instanced(amount);
	}

	cpu_ssbo::simulation_parameters cpu_ssbo::get_simulation_parameters()
	{
		return sim_params;
	}

	void cpu_ssbo::set_dynamic_simulation_parameters(simulation_parameters::dynamic_parameters new_dyn_params)
	{
		sim_params.dynamic_params = new_dyn_params;
	}
}