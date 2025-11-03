import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from helpers import *
import numpy as np

class ExampleTrackball3DWindow(Trackball3DWindow):

    def __init__(self):
        super().__init__((1024, 1024), "Example Trackball3D Window")
        self.path = CameraPath([0, 1], [Camera([0, 0, 4]), Camera([2, 2, 2]).look_at([0, 0, 0])])
        self.path_t = 1

    def init_shaders(self):
        super().init_shaders()
        self.scene_op = SceneOP(self.display_res)

    def key_press(self, key):
        super().key_press(key)
        if key == glfw.KEY_C:
            self.path_t = 0
        elif key== glfw.KEY_UP:
            self.scene_op.shininess += 2.0
            print(f"Shininess value = {self.scene_op.shininess}")
        elif key== glfw.KEY_DOWN:
            self.scene_op.shininess = max(1.0, self.scene_op.shininess - 2.0)
            print(f"Shininess value = {self.scene_op.shininess}")
        elif key== glfw.KEY_I:
            self.scene_op.imp_sampling = not(self.scene_op.imp_sampling)
            print(f"Importance sampling {'enabled' if self.scene_op.imp_sampling else 'disabled'}")
        elif key== glfw.KEY_L:
            self.scene_op.reference = (self.scene_op.reference + 1)%4
            print(f"Reference {'enabled' if self.scene_op.reference else 'disabled'}")

        elif key== glfw.KEY_MINUS:
            self.scene_op.kd = max(0.0, self.scene_op.kd - 0.05)
            print(f"Diffuse constant kd = {self.scene_op.kd}")
        elif key== glfw.KEY_EQUAL:
            self.scene_op.kd = min(1.0, self.scene_op.kd + 0.05)
            print(f"Diffuse constant kd = {self.scene_op.kd}")
        elif key == glfw.KEY_X:
            self.scene_op.view_idx = (self.scene_op.view_idx + 1)%3


    def render(self):
        if self.path_t >= 1:
            camera = self.trackball.camera
        else:
            camera = self.path(self.path_t)
            self.path_t += 0.2 / self.calc_fps
        return self.scene_op.render(camera)


class SceneOP(OpenGLOP):

    def __init__(self, resolution):
        super().__init__(["shaders/plain", "shaders/latlong"], [None, None], ["shaders/conv", "shaders/latlong"], 1, resolution, create_depth_rendertargets=True)

        self.init_uniform("viewMatrix", shader_idx=0)
        self.init_uniform("projMatrix", shader_idx=0)
        self.init_uniform("modelMatrix", shader_idx=0)
        self.init_uniform("normalMatrix", shader_idx=0)
        self.init_uniform("equiMap", shader_idx=0)
        self.init_uniform("brdfMap", shader_idx=0)
        self.init_uniform("cameraPos", shader_idx=0)
        self.init_uniform("shininess", shader_idx=0)
        self.init_uniform("impSampling", shader_idx=0)
        self.init_uniform("reference", shader_idx=0)
        self.init_uniform("kd", shader_idx=0)
        self.init_uniform("brdfRes", shader_idx=0)
        self.init_uniform("vw", shader_idx=1)
        self.init_uniform("proj", shader_idx=1)
        self.init_uniform("equirectangularMap", shader_idx=1)

        self.camera_views = [[3, 3, 3], [0, 4, 0], [4, 0, 0]]
        self.view_idx = 0

        self.order = 1
        os.makedirs(rf"results_gt\order{self.order}", exist_ok=True)
        self.shininess = 8
        mesh_path = r"meshes/Stanford_bunny.ply"
        self.mesh_name = os.path.basename(mesh_path)[:-4]
        sphere_mesh = load_mesh(mesh_path)
        sphere_mesh.vertices.fit_into_cuboid(-1, 1)
        if not hasattr(sphere_mesh.vertices, "normal") or sphere_mesh.vertices.normal is None:
            o3d_mesh = sphere_mesh.open3d()
            o3d_mesh.compute_vertex_normals()
            sphere_mesh.vertices.normal = np.asarray(o3d_mesh.vertex_normals)

        self.sphere_fg = VertexArrayObject().upload_mesh(sphere_mesh, self.shaders[0])
        self.sphere_bg = VertexArrayObject().upload_mesh(sphere_mesh, self.shaders[1])
        env_path = r"..\data\envmap\large_corridor_1k.exr"
        self.model_name = os.path.basename(env_path)[:-4]
        self.equirect_map = Texture2D(load_image(env_path))
        brdf_path = rf"brdfs/diff_opt_{self.order}_13_{self.shininess}_0.0_512.exr"

        self.brdf_map = Texture2D(load_image(brdf_path))
        im_brdf = self.brdf_map.download_image()
        self.brdf_res = im_brdf.shape[0]
        print("brdf resolution", self.brdf_res)


        self.imp_sampling = False
        self.reference = 0
        self.kd = 0.0
        self.model_matrix = np.identity(4, dtype=np.float32)

    def render(self, camera):
        pos = self.camera_views[self.view_idx]
        if pos == [0, 4, 0]:
            camera = Camera(pos).look_at([0, 0, 0], up=[0, 0, -1])  # top-down view fix
        else:
            camera = Camera(pos).look_at([0, 0, 0])


        rt = self.rendertargets[0]
        w, h = rt.color.resolution

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt.color.handle, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, rt.depth.handle, 0)

        glViewport(0, 0, w, h)
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(self.shaders[0])
        glProgramUniform1i(self.shaders[0], self.uniforms["equiMap"], 0)
        glProgramUniform1i(self.shaders[0], self.uniforms["brdfMap"], 1)
        glProgramUniform1i(self.shaders[0], self.uniforms["impSampling"], int(self.imp_sampling))
        glProgramUniform1i(self.shaders[0], self.uniforms["reference"], int(self.reference))
        glProgramUniform1f(self.shaders[0], self.uniforms["shininess"], self.shininess)
        glProgramUniform1f(self.shaders[0], self.uniforms["kd"], self.kd)
        glProgramUniform1f(self.shaders[0], self.uniforms["brdfRes"], int(self.brdf_res))



        view_matrix = camera.view_matrix()
        proj_matrix = camera.projection_matrix(w / h, 0.1, 100.0)
        normal_matrix = np.linalg.inv(self.model_matrix[:3, :3]).transpose()
        
        glProgramUniformMatrix4fv(self.shaders[0], self.uniforms["viewMatrix"], 1, GL_TRUE, view_matrix)
        glProgramUniformMatrix4fv(self.shaders[0], self.uniforms["projMatrix"], 1, GL_TRUE, proj_matrix)
        glProgramUniformMatrix4fv(self.shaders[0], self.uniforms["modelMatrix"], 1, GL_TRUE, self.model_matrix)
        glProgramUniformMatrix3fv(self.shaders[0], self.uniforms["normalMatrix"], 1, GL_TRUE, normal_matrix)
        
        glProgramUniform3fv(self.shaders[0], self.uniforms["cameraPos"], 1, camera.position)

        glActiveTexture(GL_TEXTURE0)
        self.equirect_map.set_params(min_filter=GL_LINEAR, mag_filter=GL_LINEAR, wrap=GL_CLAMP_TO_BORDER)
        
        glActiveTexture(GL_TEXTURE1)
        self.brdf_map.set_params(min_filter=GL_LINEAR, mag_filter=GL_LINEAR, wrap=GL_CLAMP_TO_BORDER)
        
        self.sphere_fg.draw()
        self.equirect_map.unbind()
        self.brdf_map.unbind()
        
        save_path = os.path.join(rf"results_gt\order{self.order}", f"{self.model_name}_{self.shininess}_{self.mesh_name}_view{self.view_idx}.exr")
        img = rt.color.download_image()
        print("img.shape", img.shape)
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return rt.color

if __name__ == "__main__":
    window = ExampleTrackball3DWindow()
    window.run()