// Vertex shader
// The output of a vertex shader is the position of the vertex
// after any kind of elaboration
// #version 410 core

layout (location = 0) in vec3 position;

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;

out gl_PerVertex { vec4 gl_Position; };

void main()
{
	vec4 modelview_position = view_matrix * model_matrix * vec4(position, 1);
	gl_Position = projection_matrix * modelview_position;
}