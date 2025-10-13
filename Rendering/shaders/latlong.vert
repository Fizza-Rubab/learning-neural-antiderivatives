#version 420 core

in vec3 position;

out vec3 WorldPos;

uniform mat4 vw;
uniform mat4 proj;

void main() {
    WorldPos = position;
    vec4 pos = proj * mat4(mat3(vw)) * vec4(position, 1.0);
    gl_Position = pos.xyww;
}
