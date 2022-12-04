#include "cpu_ssbo.h"

// std libraries
#include <vector>
#include <math.h>

#include <glad.h>
#include <glm/glm.hpp>

// utils libraries
#include "../utils/utils.h"
#include "../utils/CUDA/cuda_utils.h"
#include "boid_behaviours.h"

namespace utils::runners
{
	cpu_ssbo::cpu_ssbo(simulation_parameters params) :
		ssbo_runner{ {"shaders/ssbo.vert", "shaders/basic.frag"}, params },
		amount{ sim_params.boid_amount },
		positions { std::vector<float4>(amount) },
		velocities{ std::vector<float4>(amount) }
	{
		float spawn_range = sim_params.cube_size * 0.5f;
		utils::cuda::containers::random_vec4_fill_hptr(positions.data(), amount, -spawn_range, spawn_range);
		utils::cuda::containers::random_vec4_fill_hptr(velocities.data(), amount, -1, 1);

		setup_buffer_object(ssbo_positions , GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 0, positions.data());
		setup_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 1, velocities.data());
	}

	void cpu_ssbo::naive_calculation(const float delta_time)
	{
		float4 accel_blend;
		float cs = sim_params.cube_size;
		for (size_t i = 0; i < amount; i++)
		{
			accel_blend = sim_params.alignment_coeff * behaviours::cpu::naive::alignment(i, positions.data(), velocities.data(), amount, sim_params.boid_fov)
				+ sim_params.cohesion_coeff * behaviours::cpu::naive::cohesion(i, positions.data(), amount, sim_params.boid_fov)
				+ sim_params.separation_coeff * behaviours::cpu::naive::separation(i, positions.data(), amount, sim_params.boid_fov)
				+ sim_params.wall_separation_coeff * behaviours::cpu::naive::wall_separation(i, positions.data(), sim_volume.data(), amount);

			//velocities[i] = normalize(velocities[i]) + normalize(accel_blend) * delta_time; //v = u + at
			velocities[i] = utils::math::normalize_or_zero(velocities[i] + accel_blend * delta_time); //v = u + at
			positions[i] += velocities[i] * sim_params.boid_speed * delta_time; //s = vt
			positions[i] = clamp(positions[i], { -cs,-cs,-cs,0 }, { cs,cs,cs,0 }); // ensures boids remain into the cube
		}

		update_buffer_object(ssbo_positions , GL_SHADER_STORAGE_BUFFER, 0, sizeof(float4), amount, positions.data());
		update_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, 0, sizeof(float4), amount, velocities.data());
	}

	void cpu_ssbo::uniform_grid_calculation(const float delta_time)
	{
		// 1) crea l'array per la griglia con una certa gridresolution (number of cells per line)
		float grid_resolution = 2.f; // TODO move grid_resolution or cell_size to simulation_parameters
		float cell_size = sim_params.cube_size/grid_resolution;
		float cube_half_size = sim_params.cube_size / 2;
		std::vector<int> boid_cell_indices(amount);
		// 2) data la loro posizione, affibbia un indice di griglia a ogni boid(indice 3d{ i,j,k } che però può e deve essere linearizzato)
		int x, y, z, linear_index;
		for (size_t i = 0; i < amount; i++)
		{
			x = utils::math::normalized_value_in_range(positions[i].x, -cube_half_size, cube_half_size) * grid_resolution;
			y = utils::math::normalized_value_in_range(positions[i].y, -cube_half_size, cube_half_size) * grid_resolution;
			z = utils::math::normalized_value_in_range(positions[i].z, -cube_half_size, cube_half_size) * grid_resolution;
			linear_index = x * grid_resolution * grid_resolution + y * grid_resolution + z;
			boid_cell_indices[i] = linear_index;
		}
		// 3) ordina l'array di boid secondo la loro posizione di griglia(scattered=ordina solo il boid index, coherent = ordina pure velocities e positions)
		std::sort(boid_cell_indices.begin(), boid_cell_indices.end(), [](int a, int b) -> bool {return a < b; });
		// 4) calcola "start" e "end" di ogni cella della griglia(ovvero il range di indici uguali dei boid adiacenti)
		
		// 5) calcola le velocità usando solo quella cella come neighborhood(o al peggio checka le 8 adiacenti(? )(da controllare))
	}

	void cpu_ssbo::calculate(const float delta_time)
	{
		//naive_calculation(delta_time);
		uniform_grid_calculation(delta_time);
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

	void cpu_ssbo::set_simulation_parameters(cpu_ssbo::simulation_parameters new_params)
	{
		sim_params = new_params;
	}
}