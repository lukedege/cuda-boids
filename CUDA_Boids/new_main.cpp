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

#include "runners/gpu_angle_based.h"
#include "runners/gpu_vel_based.h"
#include "runners/cpu_angle_based.h"
#include "runners/cpu_vel_based.h"

namespace chk = utils::cuda::checks;
namespace ugo = utils::graphics::opengl;

int main()
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

	// Runner setup
	size_t amount = 1024000;
	utils::runners::gpu_vel_based spec_runner{ triangle_mesh, amount };
	utils::runners::boid_runner* runner = &spec_runner;

	// Camera setup
	ugo::Camera camera{ glm::vec3(0, 0, 50), GL_TRUE };
	glm::mat4 projection_matrix = glm::perspective(45.0f, (float)window_size.first / (float)window_size.second, 0.1f, 10000.0f);
	glm::mat4 view_matrix = glm::mat4(1);

	GLfloat delta_time = 0.0f, last_frame = 0.0f, current_fps = 0.0f;
	GLfloat avg_calc = 1.f, avg_fps = 1.f;
	GLfloat alpha = 0.9;
	while (wdw.is_open())
	{
		// we determine the time passed from the beginning
		// and we calculate the time difference between current frame rendering and the previous one
		GLfloat currentFrame = glfwGetTime();
		delta_time = currentFrame - last_frame;
		last_frame = currentFrame;

		view_matrix = camera.GetViewMatrix();

		// we "clear" the frame and z buffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		GLfloat before_calculations = glfwGetTime();

		runner->calculate(delta_time);

		GLfloat after_calculations = glfwGetTime();
		GLfloat delta_calculations = (after_calculations - before_calculations) * 1000; //in ms
		std::cout << "Calcs: " << delta_calculations << "ms | ";

		runner->draw(view_matrix, projection_matrix);

		current_fps = (1 / delta_time);
		std::cout << "FPS: " << current_fps << "\n";
		avg_fps  = alpha * avg_fps  + (1.0 - alpha) * (1 / delta_time);
		avg_calc = alpha * avg_calc + (1.0 - alpha) * (delta_calculations);

		glfwSwapBuffers(glfw_window);
		glfwPollEvents();
	}
	std::cout << "-----------------------------------\n";
	std::cout << "Average FPS  : " << avg_fps  << "\n";
	std::cout << "Average calcs: " << avg_calc << "ms \n";

	return 0;
}