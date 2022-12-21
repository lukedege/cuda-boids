// std libraries
#include <iostream>
#include <iomanip>
#include <string>

// OpenGL libraries
#ifdef _WIN32
#define APIENTRY __stdcall
#endif

#include <glad.h>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// utils libraries
#include "utils/window.h"
#include "utils/orbit_camera.h"

#include "runners/gpu_ssbo.h"
#include "runners/cpu_ssbo.h"
#include "runners/cpu_vao.h"
#include "runners/gpu_vao.h"

namespace ugl = utils::graphics::opengl;

// Input and callbacks setup
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mouse_pos_callback(GLFWwindow* window, double x_pos, double y_pos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mode);
void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void process_keys(ugl::window& window, GLfloat delta_time);
void calculate_breath_parameters(utils::runners::boid_runner::simulation_parameters& params, float speed, float amplitude);

GLfloat mouse_last_x, mouse_last_y, x_offset, y_offset;
bool left_mouse_pressed;
bool keys[1024];

ugl::orbit_camera camera{ glm::vec3(0, 0, 0), 50.f };

int main()
{
	// OpenGL window setup
	ugl::window wdw
	{
		ugl::window::window_create_info
		{
			{ "CUDA Boid simulation" }, //.title
			{ 4       }, //.gl_version_major
			{ 3       }, //.gl_version_minor
			{ 1280    }, //.window_width
			{ 720     }, //.window_height
			{ 1280    }, //.viewport_width
			{ 720     }, //.viewport_height
			{ false   }, //.resizable
			{ true   }, //.debug_gl
		}
	};

	GLFWwindow* glfw_window = wdw.get();
	ugl::window::window_size ws = wdw.get_size();
	float width = static_cast<float>(ws.width), height = static_cast<float>(ws.height);

	// Callbacks linking with glfw
	glfwSetKeyCallback(glfw_window, key_callback);
	glfwSetCursorPosCallback(glfw_window, mouse_pos_callback);
	glfwSetMouseButtonCallback(glfw_window, mouse_button_callback);
	glfwSetScrollCallback(glfw_window, mouse_scroll_callback);
	
	// Imgui setup
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(glfw_window, true);
	ImGui_ImplOpenGL3_Init("#version 430");

	// Runner setup
	utils::runners::boid_runner::simulation_parameters params
	{
		{
			{ 100000 },//boid_amount
			{ 200.f },//cube_size
			{ utils::runners::boid_runner::simulation_type::COHERENT_GRID }, // simulation_type
		},
		{
			{ 5.0f },//boid_speed
			{ 3   },//boid_fov
			{ 1.0f },//alignment_coeff
			{ 0.8f },//cohesion_coeff
			{ 1.0f },//separation_coeff
			{ 10.0f },//wall_separation_coeff
		}
	};
	bool  breath_enabled   = false;
	float breath_speed     = 1.f;
	float breath_amplitude = 0.25f;

	utils::runners::gpu_ssbo runner{ params };
	
	// Visualization matrices setup for camera
	glm::mat4 projection_matrix = glm::perspective(45.0f, width / height, 0.1f, 10000.0f);
	glm::mat4 view_matrix = glm::mat4(1);
	camera.update_distance(params.static_params.cube_size * 1.2f);

	// Measurements variables setup
	GLfloat delta_time = 0.0f, last_frame = 0.0f, current_fps = 0.0f;
	GLfloat before_calculations = 0.0f, after_calculations = 0.0f, delta_calculations = 0.0f;
	GLfloat avg_calc = 1.f, avg_fps = 1.f;
	GLfloat alpha = 0.9;
	std::cout << std::setprecision(4) << std::fixed;

	// Main loop
	while (wdw.is_open())
	{
		// Calculate the time difference between current frame rendering and the previous one
		GLfloat current_frame = glfwGetTime();
		delta_time = current_frame - last_frame;
		last_frame = current_frame;

		// Process input
		glfwPollEvents();
		process_keys(wdw, delta_time);

		// CALCULATION STEP

		before_calculations = glfwGetTime();
		runner.calculate(delta_time);
		after_calculations = glfwGetTime(); // TODO measure time for kernel using CudaEvents
		delta_calculations = (after_calculations - before_calculations) * 1000; //in ms
		
		// DRAW STEP

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		view_matrix = camera.view_matrix();
		runner.draw(view_matrix, projection_matrix); //TODO la projection matrix è fissa magari non serve aggiornarla ogni frame tbh

		// ImGUI window creation
		ImGui_ImplOpenGL3_NewFrame();// Tell OpenGL a new Imgui frame is about to begin
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Boid settings");
		ImGui::SliderFloat("Boid Speed"      , &params.dynamic_params.boid_speed           , 0, 30, "%.3f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderInt  ("Boid Fov"        , &params.dynamic_params.boid_fov             , 1, 20, "%d", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat("Alignment"       , &params.dynamic_params.alignment_coeff      , 0, 5 , "%.3f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat("Cohesion"        , &params.dynamic_params.cohesion_coeff       , 0, 5 , "%.3f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat("Separation"      , &params.dynamic_params.separation_coeff     , 0, 5 , "%.3f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat("Wall Separation" , &params.dynamic_params.wall_separation_coeff, 0, 20, "%.3f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::Checkbox   ("Breath Effect"   , &breath_enabled);
		ImGui::SliderFloat("Breath Speed"    , &breath_speed    , 0, 5, "%.3f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat("Breath Amplitude", &breath_amplitude, 0, 1, "%.3f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::End();

		// Renders the ImGUI elements
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		// Update parameters changed through imgui
		if (breath_enabled) calculate_breath_parameters(params, breath_speed, breath_amplitude);;
		runner.set_dynamic_simulation_parameters(params.dynamic_params);

		// Update performance measurements
		current_fps = (1 / delta_time);
		avg_fps = alpha * avg_fps + (1.0 - alpha) * (1 / delta_time);
		avg_calc = alpha * avg_calc + (1.0 - alpha) * (delta_calculations);
		//std::cout << "Calcs: " << delta_calculations << "ms | ";
		//std::cout << "FPS: " << current_fps << "\n";

		glfwSwapBuffers(glfw_window);
	}
	// Print results
	std::cout << "-----------------------------------\n";
	std::cout << "Average FPS  : " << avg_fps << "\n";
	std::cout << "Average calcs: " << avg_calc << "ms \n";

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	return 0;
}

void process_keys(ugl::window& window, GLfloat delta_time)
{
	if (keys[GLFW_KEY_ESCAPE])
		window.close();
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (action == GLFW_PRESS)
		keys[key] = true;
	else if (action == GLFW_RELEASE)
		keys[key] = false;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mode)
{
	ImGuiIO& io = ImGui::GetIO();
	io.AddMouseButtonEvent(button, GLFW_PRESS);

	if(!io.WantCaptureMouse)
		left_mouse_pressed  = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
}

void mouse_pos_callback(GLFWwindow* window, double xpos, double ypos) 
{
	if (left_mouse_pressed) 
	{
		int width, height;
		glfwGetWindowSize(window, &width, &height);
		float fwidth = static_cast<float>(width), fheight = static_cast<float>(height);
		float phi = camera.get_phi(), theta = camera.get_theta();

		// compute new camera parameters with polar (spherical) coordinates
		phi   += (xpos - mouse_last_x) / fwidth ;
		theta -= (ypos - mouse_last_y) / fheight;
		theta = std::clamp(theta, 0.01f, 3.14f);
		camera.update_angle(phi, theta);
	}

	mouse_last_x = xpos;
	mouse_last_y = ypos;
}

void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	float zoom = camera.get_distance();
	zoom -= yoffset;
	zoom = std::clamp(zoom, 0.1f, 1000.0f);
	camera.update_distance(zoom);
}

void calculate_breath_parameters(utils::runners::boid_runner::simulation_parameters& params, float speed, float amplitude)
{
	float time = glfwGetTime();
	params.dynamic_params.cohesion_coeff   = 1.f - ((cos(time * speed) + 1) * amplitude / 2.f);
	params.dynamic_params.separation_coeff = 1.f - ((sin(time * speed) + 1) * amplitude / 2.f);
}