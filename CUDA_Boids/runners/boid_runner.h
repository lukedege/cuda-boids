#pragma once
#include <glm/glm.hpp>
#include <array>

#include "../utils/utils.h"
#include "../utils/shader.h"
#include "../utils/mesh.h"

namespace utils::runners
{
	//all data should be in unified memory or both host and device memory
	class boid_runner 
	{
	public:
		struct simulation_parameters
		{
			size_t boid_amount          { 50 };
			float  boid_speed           { 3.0f };
			float  boid_fov             { 10.f };
			float  alignment_coeff      { 1.0f };
			float  cohesion_coeff       { 0.9f };
			float  separation_coeff     { 0.5f };
			float  wall_separation_coeff{ 3.0f };
			float  cube_size            { 20.f };
		};

		virtual void calculate(const float delta_time) = 0;
		virtual void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix) = 0;

		virtual simulation_parameters get_simulation_parameters() = 0;
		virtual void set_simulation_parameters(simulation_parameters new_params) = 0;

	protected:
		boid_runner(utils::graphics::opengl::Shader&& boid_shader, simulation_parameters params = {}) :
			sim_params{ params },
			sim_volume{ reset_volume() },
			cube_mesh{ reset_cube_mesh() },
			boid_shader{ std::move(boid_shader) },
			debug_shader{ "shaders/mvp.vert", "shaders/basic.frag" }
		{
			setup_buffer_object(ubo_matrices, GL_UNIFORM_BUFFER, sizeof(glm::mat4), 2, 2, 0);
		}

		inline std::array<utils::math::plane, 6> reset_volume()
		{
			float val = sim_params.cube_size * 0.5f;
			return
			{
				utils::math::plane{{  val,0,0,1 }, { -1,0,0,0 }}, //xp
				utils::math::plane{{ -val,0,0,1 }, {  1,0,0,0 }}, //xm
				utils::math::plane{{ 0, val,0,1 }, { 0,-1,0,0 }}, //yp
				utils::math::plane{{ 0,-val,0,1 }, { 0, 1,0,0 }}, //ym
				utils::math::plane{{ 0,0, val,1 }, { 0,0,-1,0 }}, //zp
				utils::math::plane{{ 0,0,-val,1 }, { 0,0, 1,0 }}, //zm
			};
		}
		
		inline utils::graphics::opengl::Mesh reset_cube_mesh()
		{
			float val = sim_params.cube_size * 0.5f;
			std::vector<utils::graphics::opengl::Vertex> vertices
			{
				{{ -val,  val, -val, }},//0 up   front sx
				{{  val,  val, -val, }},//1 up   front dx
				{{  val, -val, -val, }},//2 down front dx
				{{ -val, -val, -val, }},//3 down front sx
				{{ -val,  val,  val, }},//4 up   back  sx
				{{  val,  val,  val, }},//5 up   back  dx
				{{  val, -val,  val, }},//6 down back  dx
				{{ -val, -val,  val, }},//7 down back  sx
			};
			std::vector<GLuint>  indices
			{
				// front face
				0, 1, 1, 2, 2, 3, 3, 0,
				// back face
				4, 5, 5, 6, 6, 7, 7, 4,
				// links
				0, 4, 1, 5, 2, 6, 3, 7
			};
			return { vertices, indices };
		}

		inline void setup_buffer_object(GLuint& buffer_object, GLenum target, size_t element_size, size_t element_amount, int bind_index, void* data)
		{
			glGenBuffers(1, &buffer_object);
			glBindBuffer(target, buffer_object);

			size_t alloc_size = element_size * element_amount;
			glBufferData(target, alloc_size, NULL, GL_DYNAMIC_DRAW); // allocate alloc_size bytes of memory
			glBindBufferBase(target, bind_index, buffer_object);

			if (data != 0)
				glBufferSubData(target, 0, alloc_size, data);        // fill buffer object with data

			glBindBuffer(target, 0);
		}

		inline void update_buffer_object(GLuint& buffer_object, GLenum target, size_t offset, size_t element_size, size_t element_amount, void* data)
		{
			glBindBuffer(target, buffer_object);
			glBufferSubData(target, offset, element_amount * element_size, data);
			glBindBuffer(target, 0);
		}

		// Simulation related data
		simulation_parameters sim_params;
		std::array<utils::math::plane,6> sim_volume;

		// Debug and visualization related data
		utils::graphics::opengl::Mesh cube_mesh;
		utils::graphics::opengl::Shader boid_shader;
		utils::graphics::opengl::Shader debug_shader;

		GLuint ubo_matrices; // for camera's projection and view matrices
	};
}