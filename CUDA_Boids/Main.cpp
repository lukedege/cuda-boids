// std libraries
#include <iostream>
#include <iomanip>
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

#include "runners/gpu_ssbo.h"
#include "runners/cpu_ssbo.h"
#include "runners/cpu_vao.h"
#include "runners/gpu_vao.h"

namespace chk = utils::cuda::checks;
namespace ugo = utils::graphics::opengl;

// input stuff TODO: MOVE SOMEWHERE ELSE
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mouse_pos_callback(GLFWwindow* window, double x_pos, double y_pos);
void process_camera_input(ugo::Camera& cam, GLfloat delta_time);

GLfloat mouse_last_x, mouse_last_y, x_offset, y_offset;
bool first_mouse = true;

bool keys[1024];

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
			{ true   }, //.debug_gl
		}
	};

	GLFWwindow* glfw_window = wdw.get();
	auto window_size = wdw.get_size();

	// setup callbacks
	glfwSetKeyCallback(glfw_window, key_callback);
	//glfwSetCursorPosCallback(glfw_window, mouse_pos_callback);

	// we disable the mouse cursor
	//glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	
	// Runner setup
	utils::runners::boid_runner::simulation_parameters params
	{
		{ 50   },//boid_amount
		{ 5.0f },//boid_speed
		{ 5.f  },//boid_fov
		{ 1.0f },//alignment_coeff
		{ 0.8f },//cohesion_coeff
		{ 1.0f },//separation_coeff
		{ 3.0f },//wall_separation_coeff
		{ 10.f },//cube_size
	};
	utils::runners::boid_runner* runner;
	runner->simulation_params = params;

	utils::runners::cpu_vel_ssbo spec_runner;
	runner = &spec_runner;
	
	// Camera setup
	ugo::Camera camera{ glm::vec3(0, 0, 50), GL_FALSE };
	glm::mat4 projection_matrix = glm::perspective(45.0f, (float)window_size.first / (float)window_size.second, 0.1f, 10000.0f);
	glm::mat4 view_matrix = glm::mat4(1);

	GLfloat delta_time = 0.0f, last_frame = 0.0f, current_fps = 0.0f;
	GLfloat before_calculations = 0.0f, after_calculations = 0.0f, delta_calculations = 0.0f;
	GLfloat avg_calc = 1.f, avg_fps = 1.f;
	GLfloat alpha = 0.9;
	std::cout << std::setprecision(4) << std::fixed;
	while (wdw.is_open())
	{
		// we determine the time passed from the beginning
		// and we calculate the time difference between current frame rendering and the previous one
		GLfloat currentFrame = glfwGetTime();
		delta_time = currentFrame - last_frame;
		last_frame = currentFrame;

		glfwPollEvents();
		process_camera_input(camera, delta_time);
		view_matrix = camera.GetViewMatrix();

		// we "clear" the frame and z buffer
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		before_calculations = glfwGetTime();

		runner->calculate(delta_time);

		after_calculations = glfwGetTime();
		delta_calculations = (after_calculations - before_calculations) * 1000; //in ms

		runner->draw(view_matrix, projection_matrix);

		current_fps = (1 / delta_time);
		//std::cout << "Calcs: " << delta_calculations << "ms | ";
		//std::cout << "FPS: " << current_fps << "\n";
		avg_fps = alpha * avg_fps + (1.0 - alpha) * (1 / delta_time);
		avg_calc = alpha * avg_calc + (1.0 - alpha) * (delta_calculations);

		glfwSwapBuffers(glfw_window);
	}
	std::cout << "-----------------------------------\n";
	std::cout << "Average FPS  : " << avg_fps << "\n";
	std::cout << "Average calcs: " << avg_calc << "ms \n";

	return 0;
}

void process_camera_input(ugo::Camera& cam, GLfloat delta_time)
{
	cam.ProcessMouseMovement(x_offset, y_offset);
	x_offset = 0; y_offset = 0;
	if (keys[GLFW_KEY_Q])
		cam.ProcessKeyboard(ugo::Camera::Directions::BACKWARD, delta_time);
	if (keys[GLFW_KEY_E])
		cam.ProcessKeyboard(ugo::Camera::Directions::FORWARD, delta_time);
	if (keys[GLFW_KEY_A])
		cam.ProcessKeyboard(ugo::Camera::Directions::LEFT, delta_time);
	if (keys[GLFW_KEY_D])
		cam.ProcessKeyboard(ugo::Camera::Directions::RIGHT, delta_time);
	if (keys[GLFW_KEY_W])
		cam.ProcessKeyboard(ugo::Camera::Directions::UP, delta_time);
	if (keys[GLFW_KEY_S])
		cam.ProcessKeyboard(ugo::Camera::Directions::DOWN, delta_time);
}
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (action == GLFW_PRESS)
		keys[key] = true;
	else if (action == GLFW_RELEASE)
		keys[key] = false;
}
void mouse_pos_callback(GLFWwindow* window, double x_pos, double y_pos)
{
	if (first_mouse)
	{
		mouse_last_x = x_pos;
		mouse_last_y = y_pos;
		first_mouse = false;
	}

	x_offset = x_pos - mouse_last_x;
	y_offset = mouse_last_y - y_pos;

	mouse_last_x = x_pos;
	mouse_last_y = y_pos;
}