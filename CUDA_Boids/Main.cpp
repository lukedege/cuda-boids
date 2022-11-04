// std libraries
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

// OpenGL libraries
#ifdef _WIN32
#define APIENTRY __stdcall
#endif

#include <glad.h>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp> 

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

// utils libraries
#include "utils/CUDA/vector_math.h"
#include "utils/CUDA/cuda_utils.cuh"
#include "utils/CUDA/cudaGLmanager.h"
#include "utils/utils.h"
#include "utils/window.h"
#include "utils/shader.h"
#include "utils/mesh.h"
#include "utils/entity.h"
#include "utils/flock.h"
#include "utils/camera.h"

namespace chk = utils::cuda::checks;
namespace ugo = utils::graphics::opengl;


__global__ void test_kernel(float2* ssbo_positions, float* ssbo_angles, size_t size, float delta_time)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	float2 vel{ 1,1 };
	float2 vel_norm{ normalize(vel) };
	float2 ref{ 1,0 };
	//printf("%f, %f, %f\n", delta_time, ssbo_positions[i].x, ssbo_positions[i].y);
	if (i < size)
	{
		ssbo_angles[i] = acos(clamp(dot(vel_norm, ref), -1.f, 1.f));
		ssbo_positions[i] += vel * delta_time;
	}
}

__global__ void test_kernel(float2* ssbo_positions, float2* ssbo_velocities, size_t size, float delta_time)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float2 vel{ 1,1 };
	//printf("%f, %f, %f\n", delta_time, ssbo_positions[i].x, ssbo_positions[i].y);
	if (i < size)
	{
		ssbo_velocities[i] = vel;
		ssbo_positions[i] += vel * delta_time;
	}
}

__host__ void test_cpu(glm::vec2* positions, float* angles, GLuint ssbo_positions, GLuint ssbo_angles, size_t size, float delta_time)
{
	glm::vec2 vel{ 1,1 };
	glm::vec2 vel_norm{ glm::normalize(vel) };
	glm::vec2 x{ 1,0 };

	for (size_t i = 0; i < size; i++)
	{
		angles[i] = glm::angle(vel_norm, x);
		positions[i] += vel * delta_time;
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_positions);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size * sizeof(glm::vec2), positions);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_angles);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size * sizeof(float), angles);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

int mainz()
{
	ugo::window wdw
	{
		ugo::window::window_create_info
		{
			{ "Prova" }, //.title
			{ 4       }, //.gl_version_major
			{ 3       }, //.gl_version_minor
			{ 1200    }, //.window_width
			{ 900     }, //.window_height
			{ 1200    }, //.viewport_width
			{ 900     }, //.viewport_height
			{ false   }, //.resizable
			{ false   }, //.debug_gl
		}
	};

	GLFWwindow* glfw_window = wdw.get();
	auto window_size = wdw.get_size();

	ugo::Shader basic_shader{ "shaders/ssbo_instanced_angle.vert", "shaders/basic.frag", {}, 4, 3 };

	std::vector<ugo::Vertex> vertices
	{
		{{  0.0f,  0.5f, 0.0f,}},// top right
		{{  0.5f, -0.5f, 0.0f }},// bottom right
		{{ -0.5f, -0.5f, 0.0f }},// bottom left
	};
	std::vector<GLuint> indices
	{
		0, 1, 2,   // first triangle
	};

	ugo::Mesh   triangle_mesh{ vertices, indices };
	
	
	GLuint ssbo_positions; // shader_storage_buffer_object
	GLuint ssbo_angles; // shader_storage_buffer_object

	size_t amount = 1024;

	int pos_alloc_size = sizeof(glm::vec2) * amount;
	int ang_alloc_size = sizeof(float) * amount;

	size_t block_size = 32;
	size_t grid_size = utils::cuda::math::ceil(amount, block_size);

	ugo::Flock triangles{ triangle_mesh, amount };
	std::vector<float> angles (amount);

	utils::containers::random_vec2_fill_cpu(triangles.positions, -20, 20);
	
	utils::gl::setup_ssbo(ssbo_positions, pos_alloc_size, 0, triangles.positions.data());
	utils::gl::setup_ssbo(ssbo_angles   , ang_alloc_size, 1, 0);

	utils::cuda::gl_manager cuda_gl_manager;

	float2* ssbo_positions_dptr = (float2*) cuda_gl_manager.add_resource(ssbo_positions, cudaGraphicsMapFlagsNone);
	float * ssbo_angles_dptr    = (float *) cuda_gl_manager.add_resource(ssbo_angles   , cudaGraphicsMapFlagsNone);

	// Camera setup
	ugo::Camera camera{ glm::vec3(0, 0, 50), GL_TRUE };
	glm::mat4 projection_matrix = glm::perspective(45.0f, (float)window_size.first / (float)window_size.second, 0.1f, 10000.0f);
	glm::mat4 view_matrix = glm::mat4(1);

	GLfloat delta_time = 0.0f;
	GLfloat last_frame = 0.0f;
	while (wdw.is_open())
	{
		// we determine the time passed from the beginning
		// and we calculate the time difference between current frame rendering and the previous one
		GLfloat currentFrame = glfwGetTime();
		delta_time = currentFrame - last_frame;
		last_frame = currentFrame;

		view_matrix = camera.GetViewMatrix();
		basic_shader.use();
		basic_shader.setMat4("view_matrix", view_matrix);
		basic_shader.setMat4("projection_matrix", projection_matrix);

		// we "clear" the frame and z buffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		GLfloat before_calculations = glfwGetTime();

		test_kernel CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_angles_dptr, amount, delta_time);
		cudaDeviceSynchronize();

		//test_cpu(triangles.positions.data(), angles.data(), ssbo_positions, ssbo_angles, amount, delta_time);

		GLfloat after_calculations = glfwGetTime();
		GLfloat delta_calculations = (after_calculations - before_calculations) ;
		std::cout << "Calcs: " << delta_calculations * 1000 << "ms | ";

		triangles.draw(basic_shader, view_matrix);

		std::cout << "FPS: " << (1 / delta_time) << "\n";
		glfwSwapBuffers(glfw_window);
		glfwPollEvents();
	}


	return 0;
}