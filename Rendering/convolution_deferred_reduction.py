import os
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from helpers import *
from network import *    


class ExampleTrackball3DWindow(Trackball3DWindow):
    def __init__(self):
        super().__init__((1024, 1024), "Convolution outside shader")
        self.path = CameraPath([0, 1], [Camera([0, 0, 4]), Camera([2, 2, 2]).look_at([0, 0, 0])])
        self.path_t = 1
        self.init_shaders()

    def init_shaders(self):
        super().init_shaders()
        self.scene_op = SceneOP(self.display_res)

    def key_press(self, key):
        super().key_press(key)
        if key == glfw.KEY_C:
            self.path_t = 0
        if key == glfw.KEY_X:
            self.scene_op.view_idx = (self.scene_op.view_idx + 1)%3

    def render(self):
        if self.path_t >= 1:
            camera = self.trackball.camera
        else:
            camera = self.path(self.path_t)
            self.path_t += 0.2 / self.calc_fps
        try:
            result = self.scene_op.render(camera)
            return result
        except Exception as e:
            print(f"Error during scene rendering: {e}")
            return None

class SceneOP(OpenGLOP):
    def __init__(self, resolution):
        super().__init__(["shaders/plain"], [None], ["shaders/extract_info"], 1, resolution, create_depth_rendertargets=True)

        self.init_uniform("viewMatrix", shader_idx=0)
        self.init_uniform("projMatrix", shader_idx=0)
        self.init_uniform("modelMatrix", shader_idx=0)
        self.init_uniform("normalMatrix", shader_idx=0)
        self.init_uniform("cameraPos", shader_idx=0)
        self.init_uniform("shininess", shader_idx=0)

        self.camera_views = [[3, 3, 3], [0, 4, 0], [4, 0, 0]]
        self.view_idx = 0

        self.model_matrix = np.identity(4, dtype=np.float32)
        mesh_path = "meshes/FAUST_tr_reg_077.ply"
        self.mesh_name = os.path.basename(mesh_path)[:-4]
        mesh = load_mesh(mesh_path)
        mesh.vertices.fit_into_cuboid(-1, 1)
        if not hasattr(mesh.vertices, "normal") or mesh.vertices.normal is None:
            o3d_mesh = mesh.open3d()
            o3d_mesh.compute_vertex_normals()
            mesh.vertices.normal = np.asarray(o3d_mesh.vertex_normals)
        self.vao = VertexArrayObject().upload_mesh(mesh, self.shaders[0])

        res = resolution
        self.tex_theta = Texture2D(); self.tex_theta.allocate_memory(res, channels=1)
        self.tex_phi = Texture2D(); self.tex_phi.allocate_memory(res, channels=3)
        self.tex_shininess = Texture2D(); self.tex_shininess.allocate_memory(res, channels=1)
        self.output_textures = [self.tex_theta, self.tex_phi, self.tex_shininess]

        self.fbo = glGenFramebuffers(1)

        self.shininess = 64
        self.order = 1
        os.makedirs(rf"results\Reduction\order{self.order}", exist_ok=True)
        kernel_dict = torch.load(rf"brdfs\diff_b_grid_{self.order}_13_{self.shininess}_0.0.pth", weights_only=False)
        ctrl_pts = kernel_dict["ctrl_pts"].reshape(-1, 2).detach().cpu().numpy()
        ckpt = kernel_dict["ckpt"].reshape(-1).cpu().detach().numpy()
        self.num_diracs = ctrl_pts.shape[0]

        self.d_phi_torch = torch.from_numpy(ctrl_pts[:, 0]).view(-1, 1, 1).float().cuda()
        self.d_theta_torch = torch.from_numpy(ctrl_pts[:, 1]).view(-1, 1, 1).float().cuda()
        self.weights_torch = torch.from_numpy(ckpt).view(-1, 1, 1).float().cuda()

        env = load_image("../data/envmap/dikhololo_night_1k.exr").astype(np.float64)
        pad_frac = 0.3
        H, W, _ = env.shape
        pad_h, pad_w = int(H * pad_frac), int(W * pad_frac)
        left = env[:, :pad_w, :]; right = env[:, -pad_w:, :]
        env_padded = np.concatenate([right, env, left], axis=1)
        top = np.flip(env_padded[:pad_h], axis=0)
        bottom = np.flip(env_padded[-pad_h:], axis=0)
        env_padded = np.concatenate([top, env_padded, bottom], axis=0)
        dx = np.pi*2 / env.shape[1]
        dy = np.pi / env.shape[0]
        env_sat = np.cumsum(np.cumsum(np.cumsum(np.cumsum(env_padded, axis=0), axis=1), axis=0), axis=1) * dx**2 * dy**2
        self.env_sat_torch = torch.from_numpy(env_sat).permute(2, 0, 1).unsqueeze(0).float().cuda()
        self.pad_fraction = pad_frac
        self.u_start = self.pad_fraction / (1 + 2 * self.pad_fraction)
        self.u_end = (1 + self.pad_fraction) / (1 + 2 * self.pad_fraction)
        self.v_start = self.u_start
        self.v_end = self.u_end
        self.u_scale = self.u_end - self.u_start
        self.v_scale = self.v_end - self.v_start

        # 1.1 calibration
        if self.order==0:
            self.weight_scale = (np.pi**2) ** (self.order + 1)
        else:
            self.weight_scale = 1.4 * (np.pi**2) ** (self.order + 1)


        self.cached_theta = None
        self.cached_phi = None
        self.cached_shininess = None
        # envmaps\dikhololo_night_1k.exr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = rf"..\models\envmodels\Reduction\Reduction_dikhololo_night_1k_order={self.order+1}.pth"
        self.model_name = os.path.basename(model_path)[:-4]
        weights = torch.load(model_path)
        self.img_siren_f = CoordinateNet_ordinary(weights['output'],
                            weights['activation'],
                            weights['input'],
                            weights['channels'],
                            weights['layers'],
                            weights['encodings'],
                            weights['normalize_pe'],
                            weights["pe"],
                            norm_exp=0).cuda()
        self.img_siren_f.load_state_dict(weights['ckpt'])
        self.img_siren_f = self.img_siren_f.eval()
        self.img_siren_f.to(self.device)
        self.theta_tex2tensor = CopyTextureToTensor(self.tex_theta)
        self.phi_tex2tensor = CopyTextureToTensor(self.tex_phi)
        self.shininess_tex2tensor = CopyTextureToTensor(self.tex_shininess)


    def render(self, camera):
        pos = self.camera_views[self.view_idx]
        if pos == [0, 4, 0]:
            camera = Camera(pos).look_at([0, 0, 0], up=[0, 0, -1])  # top-down view fix
        else:
            camera = Camera(pos).look_at([0, 0, 0])

        rt = self.rendertargets[0]
        w, h = rt.color.resolution
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        attachments = [
            (GL_COLOR_ATTACHMENT0, self.tex_theta),
            (GL_COLOR_ATTACHMENT1, self.tex_phi),
            (GL_COLOR_ATTACHMENT5, self.tex_shininess)
        ]
        glDrawBuffers(len(attachments), np.array([a[0] for a in attachments], dtype=np.uint32))
        for attachment, texture in attachments:
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.handle, 0)

        if hasattr(rt, "depth") and rt.depth is not None:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, rt.depth.handle, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
            return rt.color

        glViewport(0, 0, w, h)
        glClearColor(0, 0, 0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(self.shaders[0])
        view_matrix = camera.view_matrix()
        proj_matrix = camera.projection_matrix(w / h, 0.1, 100.0)
        normal_matrix = np.linalg.inv(self.model_matrix[:3, :3]).T
        glProgramUniformMatrix4fv(self.shaders[0], self.uniforms["viewMatrix"], 1, GL_TRUE, view_matrix)
        glProgramUniformMatrix4fv(self.shaders[0], self.uniforms["projMatrix"], 1, GL_TRUE, proj_matrix)
        glProgramUniformMatrix4fv(self.shaders[0], self.uniforms["modelMatrix"], 1, GL_TRUE, self.model_matrix)
        glProgramUniformMatrix3fv(self.shaders[0], self.uniforms["normalMatrix"], 1, GL_TRUE, normal_matrix)
        glProgramUniform3fv(self.shaders[0], self.uniforms["cameraPos"], 1, camera.position)
        glProgramUniform1f(self.shaders[0], self.uniforms["shininess"], self.shininess)
        self.vao.draw()
        glUseProgram(0)
        
        theta = self.theta_tex2tensor.copy_to_tensor()[0, 0]
        phi = self.phi_tex2tensor.copy_to_tensor()[0, 0]
        shininess = self.shininess_tex2tensor.copy_to_tensor()[0, 0]

        mask = shininess > 0
        if not mask.any():
            glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
            return rt.color

        valid_idx = mask.nonzero(as_tuple=False)
        y, x = valid_idx[:, 0], valid_idx[:, 1]
        
        theta_shifted = theta[y, x][None] + self.d_theta_torch
        phi_shifted = phi[y, x][None] + self.d_phi_torch
        
        u = self.u_start + (phi_shifted * self.u_scale) / (2 * np.pi)
        v = self.v_start + (theta_shifted * self.v_scale) / np.pi
        uv_input = torch.stack((v, u), dim=-1).reshape(-1, 2)
        uv_input_switched = uv_input * 2 - 1  # Map from [0, 1] to [-1, 1]

        with torch.no_grad():
            preds = evaluate_network_in_chunks(self.img_siren_f, uv_input_switched, chunk_size=10_000)
            # print("preds.shape", preds.shape)
            preds = uv_input_switched[:,0:1]*uv_input_switched[:,1:]*preds[:, :3] - uv_input_switched[:,1:]*preds[:, 3:6] - uv_input_switched[:,0:1]*preds[:, 6:9] + preds[:, 9:]
            # print("preds.shape", preds.shape)
            preds = preds.view(self.num_diracs, -1, 3)
            # print("preds.shape", preds.shape)
        # print("uv_input_switched.shape", uv_input_switched.shape, preds.shape)
        # integral = uv_input_switched[:,0]*uv_input_switched[:,1]*preds[:, :, :3] - uv_input_switched[:,1]*preds[:, :, 3:6] - uv_input_switched[:,0]*preds[:, :, 6:9] + preds[:, :, 9:]
        # print("integral.shape", integral.shape)
        
        output_valid = (preds * self.weights_torch.view(-1, 1, 1)).sum(dim=0) * (self.weight_scale * shininess[y, x][:, None])
        
        output = torch.zeros((theta.shape[0], theta.shape[1], 3), dtype=output_valid.dtype, device=output_valid.device)
        output[y, x] = output_valid

        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        
        copy_tensor_to_texture(self.tex_phi, output.permute(2, 0, 1).unsqueeze(0))  # Leave as-is
        
        save_path = os.path.join(rf"results\Reduction\order{self.order}", f"{self.model_name}_{self.shininess}_{self.mesh_name}_view{self.view_idx}.exr")
        img = output.cpu().numpy()
        print("img.shape", img.shape)
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        

        return self.tex_phi

    def cleanup(self):
        # try:
        if hasattr(self, 'fbo') and self.fbo is not None:
            glDeleteFramebuffers(1, [self.fbo])
            self.fbo = None
        # except Exception as e:
        #     print(f"Error cleaning up SceneOP: {e}")
            
if __name__ == "__main__":
    window = ExampleTrackball3DWindow()
    window.run()