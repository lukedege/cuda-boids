#include <iostream>

#include <cuda_runtime.h>
#include "utils/CUDA/cuda_utils.h"
#include "utils/CUDA/vector_math.h"

#include "utils/shader.h"
#include "utils/window.h"
#include "utils/mesh.h"
#include "utils/camera.h"

#include "utils/utils.h"

namespace ugl = utils::graphics::opengl;

void opengl_main()
{
	ugl::window wdw
	{
		ugl::window::window_create_info
		{
			{ "Prova" }, //.title
			{ 4       }, //.gl_version_major
			{ 3       }, //.gl_version_minor
			{ 1200    }, //.window_width
			{ 1200     }, //.window_height
			{ 1200    }, //.viewport_width
			{ 1200     }, //.viewport_height
			{ false   }, //.resizable
			{ false   }, //.debug_gl
		}
	};

	GLFWwindow* glfw_window = wdw.get();
	ugl::window::window_size ws = wdw.get_size();
	float width = static_cast<float>(ws.width), height = static_cast<float>(ws.height);

	std::vector<glm::vec3> positions
	{
		{  0.0f,  1.0f, 0.0f,},// top right
		{  1.0f, -1.0f, 0.0f },// bottom right
		{ -1.0f, -1.0f, 0.0f },// bottom left
	};

	std::vector<glm::vec3> velocities
	{
		{ 1.f, 0.f, 0.0f,},// top right
		{ 0.f, 1.f, 0.0f },// bottom right
		{ 1.f, 1.f, 0.0f },// bottom left
	};

	GLuint VAO, VBO_positions, VBO_velocities;

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO_positions);
	glGenBuffers(1, &VBO_velocities);

	// VAO is made "active"    
	glBindVertexArray(VAO);

	// positions
	glBindBuffer(GL_ARRAY_BUFFER, VBO_positions);
	glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(glm::vec3), positions.data());
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);

	// velocities
	glBindBuffer(GL_ARRAY_BUFFER, VBO_velocities);
	glBufferData(GL_ARRAY_BUFFER, velocities.size() * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, velocities.size() * sizeof(glm::vec3), velocities.data());
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);


	utils::graphics::opengl::Shader shader{ "shaders/vao.vert", "shaders/basic.frag", "shaders/vao.geom" };

	shader.use();

	ugl::Camera camera{ glm::vec3(0, 0, 50), GL_TRUE };
	glm::mat4 projection_matrix = glm::perspective(45.0f, width / height, 0.1f, 10000.0f);
	glm::mat4 view_matrix = glm::mat4(1);

	while (wdw.is_open())
	{
		// we "clear" the frame and z buffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		for (int i = 0; i < positions.size(); i++)
		{
			positions[i] += velocities[i] * 0.0001f;
		}

		glBindBuffer(GL_ARRAY_BUFFER, VBO_positions);
		glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(glm::vec3), positions.data());

		glBindBuffer(GL_ARRAY_BUFFER, VBO_velocities);
		glBufferSubData(GL_ARRAY_BUFFER, 0, velocities.size() * sizeof(glm::vec3), velocities.data());

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		view_matrix = camera.GetViewMatrix();
		shader.setMat4("projection_matrix", projection_matrix);
		shader.setMat4("view_matrix", view_matrix);

		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, positions.size());
		glBindVertexArray(0);

		glfwSwapBuffers(glfw_window);
		glfwPollEvents();
	}
}

void plane_test()
{
	float cube_size = 10;
	float4 boid = { -1,3,9.9f,0 };
	utils::math::plane zp{ { 0,0, cube_size,0 }, { 0,0,-1,0 } };
	utils::math::plane zm{ { 0,0,-cube_size,0 }, { 0,0, 1,0 } };
	utils::math::plane xp{ {  cube_size,0,0,0 }, { -1,0,0,0 } };
	utils::math::plane xm{ { -cube_size,0,0,0 }, {  1,0,0,0 } };
	utils::math::plane yp{ { 0, cube_size,0,0 }, { 0,-1,0,0 } };
	utils::math::plane ym{ { 0,-cube_size,0,0 }, { 0, 1,0,0 } };
	std::vector<utils::math::plane> planes{ zp,zm,xp,xm,yp,ym };
	for (auto& p : planes)
	{
		float dist = utils::math::distance_point_plane(boid, p);
		std::cout << dist << " - ";
		std::cout << 1 / dist;
		std::cout << std::endl;
	}
}

namespace
{
	__device__ int simple2(int a, int b)
	{
		return a + b;
	}

	inline __host__ __device__ bool operator==(float4 a, float4 b)
	{
		return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
	}

	inline __host__ __device__ bool operator!=(float4 a, float4 b)
	{
		return !operator==(a, b);
	}

	inline __host__ __device__ float4 normalize_or_zero(float4 vec)
	{
		float4 zero{ 0 }, normalized = normalize(vec);
		float4 result[] = { zero, normalized };
		int is_valid = vec != zero;
		return result[is_valid]; // has to be 0 or 1, branchless
	}

	__global__ void simple()
	{
		float4 ret{ 0 };
		ret = normalize_or_zero(ret);
		printf("ciao %.2f", ret.x);
	}
}

/* Test per kernel gpu_ssbo grid
void gpu_ssbo::uniform_grid_calculation(const float delta_time)
{
	namespace grid_bhvr = behaviours::grid;

	float boid_fov = sim_params.dynamic_params.boid_fov;
	float cell_size = 2 * boid_fov; // we base our grid size based on the boid fov 
	float grid_resolution = sim_params.static_params.cube_size / cell_size;
	int cell_amount = grid_resolution * grid_resolution * grid_resolution;

	// ASSIGN GRID IDX
	size_t bci_size = sizeof(grid_bhvr::boid_cell_index) * amount;
	grid_bhvr::boid_cell_index* boid_cell_indices_dev = (grid_bhvr::boid_cell_index*)malloc(bci_size);
	grid_bhvr::boid_cell_index* boid_cell_indices_dptr;
	cudaMalloc(&boid_cell_indices_dptr, bci_size);
	assign_grid_indices CUDA_KERNEL(grid_size, block_size)(boid_cell_indices_dptr, ssbo_positions_dptr, amount, sim_params.static_params.cube_size, grid_resolution);
	cudaMemcpy(boid_cell_indices_dev, boid_cell_indices_dptr, bci_size, cudaMemcpyDeviceToHost);
	std::vector<grid_bhvr::boid_cell_index> test_device(boid_cell_indices_dev, boid_cell_indices_dev + amount);

	std::vector<float4> posv(amount);
	cudaMemcpy(posv.data(), ssbo_positions_dptr, sizeof(float4) * amount, cudaMemcpyDeviceToHost);
	std::vector<grid_bhvr::boid_cell_index> test_host{ assign_grid_indices(posv.data(), amount, sim_params.static_params.cube_size, grid_resolution) };

	// SORT
	//grid_bhvr::boid_cell_index* boid_cell_indices_dev = (grid_bhvr::boid_cell_index*)malloc(sizeof(behaviours::grid::boid_cell_index) * amount);
	//cudaMemcpy(boid_cell_indices_dev, boid_cell_indices_dptr, sizeof(behaviours::grid::boid_cell_index) * amount, cudaMemcpyDeviceToHost);
	//std::vector<grid_bhvr::boid_cell_index> test_device(boid_cell_indices_dev, boid_cell_indices_dev + amount);
	thrust::device_ptr<grid_bhvr::boid_cell_index> thr_bci(boid_cell_indices_dptr);
	thrust::sort(thr_bci, thr_bci + amount, order_by_cell_id());
	cudaMemcpy(boid_cell_indices_dev, boid_cell_indices_dptr, bci_size, cudaMemcpyDeviceToHost);
	std::vector<grid_bhvr::boid_cell_index> test_device_sorted(boid_cell_indices_dev, boid_cell_indices_dev + amount);


	// FIND RANGES

	size_t cir_size = sizeof(grid_bhvr::idx_range) * cell_amount;
	grid_bhvr::idx_range* cell_idx_range_dptr;
	grid_bhvr::idx_range* cell_idx_range_dev = (grid_bhvr::idx_range*)malloc(cir_size);
	cudaMalloc(&cell_idx_range_dptr, cir_size);
	find_cell_boid_range CUDA_KERNEL(grid_size, block_size)(cell_idx_range_dptr, boid_cell_indices_dptr, amount);
	cudaMemcpy(cell_idx_range_dev, cell_idx_range_dptr, cir_size, cudaMemcpyDeviceToHost);
	std::vector<grid_bhvr::idx_range> test_device_cir(cell_idx_range_dev, cell_idx_range_dev + cell_amount);

	std::vector<grid_bhvr::idx_range> test_host_cir{ find_cell_boid_range(test_device_sorted.data(), amount, grid_resolution) };
}*/


int mainz()
{
	plane_test();

	return 0;
}