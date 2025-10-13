#version 420 core
in vec3 position;
in vec3 normal;

uniform mat4 modelMatrix;
uniform mat3 normalMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;


out vec3 WorldPos;
out vec3 aNormal;

void main() {
    aNormal = normalize(normalMatrix * normal);
    gl_Position = projMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
    WorldPos =  (modelMatrix * vec4(position, 1.0)).xyz;
}
