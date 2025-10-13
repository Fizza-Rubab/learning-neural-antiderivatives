#version 430 core

layout(location = 0) out float outTheta;
layout(location = 1) out float outPhi;
layout(location = 2) out float outShininess;


in vec3 WorldPos;
in vec3 aNormal;

uniform vec3 cameraPos;
uniform float shininess;

const float PI = 3.14159265358979323846;

void main() {
    vec3 N = normalize(aNormal);
    vec3 V = normalize(cameraPos - WorldPos);
    vec3 RV = reflect(-V, N); 
    float theta = acos(RV.y);
    float phi =  PI + (atan(-RV.z, RV.x));
    outTheta = theta;
    outPhi = phi;
    outShininess = shininess;

}
