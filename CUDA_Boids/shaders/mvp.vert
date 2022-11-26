// Vertex shader
// The output of a vertex shader is the position of the vertex
// after any kind of elaboration
// #version 410 core

layout (location = 0) in vec3 pos;

out gl_PerVertex { vec4 gl_Position; };

layout (std140, binding = 2) uniform Matrices
{
	mat4 view_matrix;
	mat4 projection_matrix;
};

void main()
{
	// note that we read the multiplication from right to left
	gl_Position = projection_matrix * view_matrix * vec4(pos, 1.0);
}