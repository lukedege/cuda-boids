// std libraries
#include <iostream>
#include <vector>
#include <string>

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
#include "utils/cuda/cuda_utils.cuh"
#include "utils/utils.h"
#include "utils/window.h"
#include "utils/shader.h"
#include "utils/mesh.h"
#include "utils/entity.h"
#include "utils/flock.h"
#include "utils/camera.h"

namespace ugo = utils::graphics::opengl;

/*
* TODO FOR FLOCKING
* The entire area is divided into a dense enough, fixed(?) grid of neighbourhoods, 
*    each of which represents an area in which boids will influence each other
* We'll have 2 operations to perform the flocking operation 
* - Firstly, we will calculate for each boid which neighbourhood he belongs to
*	- To do so, each boid can just evaluate its own position in regards of the grid granularity and identify in which of the neighbourhoods he belongs to
*	- He will add himself to the list of the boids in that neighbourhood (list in SMEM preferably)
* - Secondly, we will calculate the new velocity/position for each boid of a neighbourhood, for all neighbourhoods
*	- It would be comfy to have the list of the boids in each neighbourhood in SMEM since it's a common information for each thread in a block
*/

__global__ void calculate_positions_gpu_kernel(glm::vec2* positions, float* angles, size_t size,  float delta_time)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("init vel\n");
	glm::vec2 vel{ -1, 1 };
	printf("norm vel\n");
	glm::vec2 vel_norm{ glm::normalize(vel) };
	glm::vec2 x{ 1,0 };
	//printf("%f, %f, %f\n", delta_time, positions[i].x, positions[i].y);
	if (i < size)
	{
		angles[i] = glm::angle(vel_norm, x); // equivalent to acos(glm::clamp(glm::dot(vel_norm, x), -1.f, 1.f));
		positions[i] += vel * delta_time;
		
	}
	
}

__host__ void calculate_positions_gpu_zero_pinned(size_t grid_size, size_t block_size, glm::vec2* positions, float* angles, size_t size,  float delta_time)
{
	calculate_positions_gpu_kernel CUDA_KERNEL(grid_size, block_size)(positions, angles, size, delta_time); // gpu
	cudaDeviceSynchronize(); // NEVER REMOVE THIS WHEN USING PINNED MEMORY IF YOU CARE ABOUT YOUR PC :)
}

__host__ void calculate_positions_gpu_unified(size_t grid_size, size_t block_size, glm::vec2* positions, float* angles, size_t size,  float delta_time)
{
	calculate_positions_gpu_kernel CUDA_KERNEL(grid_size, block_size)(positions, angles, size, delta_time); // gpu
	cudaDeviceSynchronize(); // same reason above :)
}

__host__ void calculate_positions_gpu_transfer(size_t grid_size, size_t block_size, glm::vec2* positions_gpu, glm::vec2* positions_cpu, float* angles, size_t size,  float delta_time)
{
	calculate_positions_gpu_kernel CUDA_KERNEL(grid_size, block_size)(positions_gpu, angles, size, delta_time); // gpu
	cudaMemcpy(positions_cpu, positions_gpu, size * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
}

__host__ void calculate_positions_cpu(glm::vec2* positions, float* angles, size_t size,  float delta_time)
{
	glm::vec2 vel{ -1, 1 };
	glm::vec2 vel_norm{ glm::normalize(vel) };
	glm::vec2 x{ 1,0 };

	for (size_t i = 0; i < size; i++)
	{
		//angles[i] = glm::angle(vel_norm, x);
		positions[i] += vel * delta_time;
	}
}

__host__ void setup_shader(ugo::Shader& shader, glm::vec2* positions, float* angles, size_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		shader.setVec2("positions[" + std::to_string(i) + "]", { positions[i].x, positions[i].y });
		shader.setFloat("angles[" + std::to_string(i) + "]", angles[i]);
	}
}


int maind()
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

	ugo::Shader basic_shader{ "shaders/mvp_instanced.vert", "shaders/basic.frag", {}, 4, 3 };

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

	size_t pos_size = 1024000;
	ugo::Flock triangles{ triangle_mesh, pos_size };

	utils::containers::random_vec2_fill_cpu(triangles.positions, -20, 20);

	// CUDA setup
	size_t alloc_size = pos_size * sizeof(glm::vec2);
	size_t block_size = 32;
	size_t grid_size = utils::cuda::math::ceil(pos_size, block_size);

	// Pinned
	glm::vec2* positions_pinned;
	cudaHostAlloc(&positions_pinned, alloc_size, cudaHostAllocMapped); ///pinned, with implicit zerocopy unified addressing, usable by both gpu and cpu
	cudaMemcpy(positions_pinned, triangles.positions.data(), alloc_size, cudaMemcpyHostToHost);

	// Unified 
	glm::vec2* positions_unified;
	cudaMallocManaged(&positions_unified, alloc_size, cudaMemAttachGlobal);
	cudaMemcpy(positions_unified, triangles.positions.data(), alloc_size, cudaMemcpyHostToHost);

	// Normal
	glm::vec2* positions_cpu;
	//positions_cpu = (glm::vec2*) malloc(alloc_size);
	//memcpy(positions_cpu, triangles.positions.data(), alloc_size);
	cudaMallocHost(&positions_cpu, alloc_size);
	cudaMemcpy(positions_cpu, triangles.positions.data(), alloc_size, cudaMemcpyHostToHost);

	glm::vec2* positions_gpu;
	cudaMalloc(&positions_gpu, alloc_size);
	cudaMemcpy(positions_gpu, triangles.positions.data(), alloc_size, cudaMemcpyHostToDevice);

	// Angles (Unified only for now)
	float* angles_unified;
	cudaMallocManaged(&angles_unified, pos_size * sizeof(float), cudaMemAttachGlobal);

	// Camera setup
	ugo::Camera camera{ glm::vec3(0, 0, 50), GL_TRUE };
	glm::mat4 projection_matrix = glm::perspective(45.0f, (float)window_size.first / (float)window_size.second, 0.1f, 10000.0f);
	glm::mat4 view_matrix = glm::mat4(1);

	GLfloat delta_time = 0.0f;
	GLfloat last_frame = 0.0f;

	int option = 3;
	glm::vec2* positions_final = nullptr;
	float* angles_final = nullptr;

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

		switch (option)
		{
		case 0:
			// Pinned gpu (calc time ~40k - 1024000)
			calculate_positions_gpu_zero_pinned(grid_size, block_size, positions_pinned, angles_unified, pos_size, delta_time);
			positions_final = positions_pinned;
			break;
		case 1:
			// Transfer gpu (calc time ~3.7k - 1024000)
			calculate_positions_gpu_transfer   (grid_size, block_size, positions_gpu, positions_cpu, angles_unified, pos_size, delta_time);
			positions_final = positions_cpu;
			break;// gpu
		case 2:
			// Cpu (calc time ~5.5k - 1024000)
			calculate_positions_cpu(positions_cpu, angles_unified, pos_size, delta_time);
			positions_final = positions_cpu;
			break;// cpu
		case 3:
			// Unified gpu (calc time ~1.1k - 1024000)
			calculate_positions_gpu_unified(grid_size, block_size, positions_unified, angles_unified, pos_size, delta_time);
			positions_final = positions_unified;
			break;// gpu
		}
		angles_final = angles_unified;

		GLfloat after_calculations = glfwGetTime();
		GLfloat delta_calculations = after_calculations - before_calculations;
		std::cout << "Calcs: " << delta_calculations * 1000000 << std::endl;

		setup_shader(basic_shader, positions_final, angles_final, pos_size);
		triangles.draw(basic_shader, view_matrix);

		glfwSwapBuffers(glfw_window);
		glfwPollEvents();
		
	}
	
	cudaFreeHost(positions_pinned);
	cudaFree(positions_gpu);
	//free(positions_cpu);
	cudaFreeHost(positions_cpu);
	return 0;
}