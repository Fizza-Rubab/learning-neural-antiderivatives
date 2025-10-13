import math
import numpy as np
import trimesh
import open3d
from trimesh.visual import ColorVisuals, TextureVisuals
from pathlib import Path
import os
import cv2
from functools import partial
from typing import overload
import torch
import enum
import glfw
import freetype
import re
import time
import tkinter as tk
from copy import deepcopy
from tkinter import filedialog
import logging
from datetime import datetime
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp
import importlib
import torch.utils.cpp_extension

class OpenGLFramework(enum.Enum):
    none = 0,
    glfw = 1,
    egl = 2

OGL_FRAMEWORK = OpenGLFramework.none

glfw.ERROR_REPORTING = False
if glfw.init():
    OGL_FRAMEWORK = OpenGLFramework.glfw
    print("[Using OpenGL with GLFW]")

else:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  
    OGL_FRAMEWORK = OpenGLFramework.egl
    print("[Using OpenGL with EGL]")

glfw.ERROR_REPORTING = True
key_mappings = {
    glfw.MOD_CONTROL: "Control",
    glfw.MOD_NUM_LOCK: "Num Lock",
    glfw.MOD_CAPS_LOCK: "Caps Lock",
    glfw.MOD_SUPER: "Super",
    glfw.MOD_ALT: "Alt",
    glfw.MOD_SHIFT: "Shift",
    glfw.KEY_UNKNOWN: "Unknown key",
    glfw.KEY_SPACE: "SpaceBar",
    glfw.KEY_APOSTROPHE: "Apostrophe",
    glfw.KEY_COMMA: "Comma",
    glfw.KEY_MINUS: "Minus",
    glfw.KEY_PERIOD: "Period",
    glfw.KEY_SLASH: "Slash",

    **{k: chr(ord('0') + (k - glfw.KEY_0)) for k in range(glfw.KEY_0, glfw.KEY_9 + 1)},  # 0-9

    glfw.KEY_SEMICOLON: "SemiColon",
    glfw.KEY_EQUAL: "Equal",

    **{k: chr(ord('A') + (k - glfw.KEY_A)) for k in range(glfw.KEY_A, glfw.KEY_Z + 1)},  # A-Z

    glfw.KEY_LEFT_BRACKET: "Left Bracket",
    glfw.KEY_BACKSLASH: "BackSlash",
    glfw.KEY_RIGHT_BRACKET: "Right Bracket",
    glfw.KEY_GRAVE_ACCENT: "Grave Accent",
    glfw.KEY_WORLD_1: "World 1",
    glfw.KEY_WORLD_2: "World 2",
    glfw.KEY_ESCAPE: "ESC",
    glfw.KEY_ENTER: "Enter",
    glfw.KEY_TAB: "Tab",
    glfw.KEY_BACKSPACE: "BackSpace",
    glfw.KEY_INSERT: "Insert",
    glfw.KEY_DELETE: "Delete",
    glfw.KEY_RIGHT: "Right Arrow",
    glfw.KEY_LEFT: "Left Arrow",
    glfw.KEY_DOWN: "Down Arrow",
    glfw.KEY_UP: "Up Arrow",
    glfw.KEY_PAGE_UP: "Page Up",
    glfw.KEY_PAGE_DOWN: "Page Down",
    glfw.KEY_HOME: "Home",
    glfw.KEY_END: "End",
    glfw.KEY_CAPS_LOCK: "Caps Lock",
    glfw.KEY_SCROLL_LOCK: "Scroll Lock",
    glfw.KEY_NUM_LOCK: "Num Lock",
    glfw.KEY_PRINT_SCREEN: "Print Screen",
    glfw.KEY_PAUSE: "Pause",

    **{k: f"F{k - glfw.KEY_F1 + 1}" for k in range(glfw.KEY_F1, glfw.KEY_F25 + 1)},  # F1-F25
    **{k: f"Keypad {k - glfw.KEY_KP_0}" for k in range(glfw.KEY_KP_0, glfw.KEY_KP_9 + 1)}, ## Keypad numbers (0-9)

    glfw.KEY_KP_DECIMAL: "Keypad Decimal",
    glfw.KEY_KP_DIVIDE: "Keypad Divide",
    glfw.KEY_KP_MULTIPLY: "Keypad Multiply",
    glfw.KEY_KP_SUBTRACT: "Keypad Subtract",
    glfw.KEY_KP_ADD: "Keypad Add",
    glfw.KEY_KP_ENTER: "Keypad Enter",
    glfw.KEY_KP_EQUAL: "Keypad Equal",
    glfw.KEY_LEFT_SHIFT: "Shift",
    glfw.KEY_LEFT_CONTROL: "Control",
    glfw.KEY_LEFT_ALT: "Alt",
    glfw.KEY_RIGHT_SHIFT: "Shift",
    glfw.KEY_RIGHT_CONTROL: "Control",
    glfw.KEY_RIGHT_ALT: "Alt",  
    glfw.KEY_MENU: "Menu",
    glfw.KEY_LAST: "Last Key"   
}


def init_egl():
    import OpenGL.EGL as egl
    import ctypes
    display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    assert display != egl.EGL_NO_DISPLAY, "Cannot access display"

    major = ctypes.c_int32()
    minor = ctypes.c_int32()
    ok = egl.eglInitialize(display, major, minor)
    assert ok, "Cannot initialize EGL"

    config_attribs = [
        egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT,
        egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT,
        egl.EGL_NONE]
    configs = (ctypes.c_int32 * 1)()
    num_configs = ctypes.c_int32()
    ok = egl.eglChooseConfig(display, config_attribs, configs, 1, num_configs) 
    assert ok
    assert num_configs.value == 1
    config = configs[0]

    surface_attribs = [
        egl.EGL_WIDTH, 1,
        egl.EGL_HEIGHT, 1,
        egl.EGL_NONE
    ]
    surface = egl.eglCreatePbufferSurface(display, config, surface_attribs)
    assert surface != egl.EGL_NO_SURFACE

    # Setup GL context.
    ok = egl.eglBindAPI(egl.EGL_OPENGL_API) 
    assert ok
    context = egl.eglCreateContext(display, config, egl.EGL_NO_CONTEXT, None) 
    assert context != egl.EGL_NO_CONTEXT
    ok = egl.eglMakeCurrent(display, surface, surface, context)
    assert ok
    return ok


if OGL_FRAMEWORK == OpenGLFramework.egl:
    if not init_egl():
        OGL_FRAMEWORK = OpenGLFramework.none

assert OGL_FRAMEWORK != OpenGLFramework.none, "Could not initialize OpenGL"

class Camera:
    """
    A single virtual camera, consisting of a position, a rotation, and a projection type with corresponding intrinsics.

    Attributes
    ----------
    position : np.ndarray
        The position of the camera as 3-element array of xyz coordinates.
    rotation : Rotation
        A rotation from the initial camera orientation (x is right, y is up, z is forward) to the desired one.
    perspective : bool
        Indicates whether this camera uses orthographic (false) or perspective (true) projection.
    sizey: float
        Height of orthographically projected box. Only present if perspective is false.
    fovy : float
        Vertical field of view. Only present if perspective is true.
    """

    def __init__(self, position=None, rotation=None, perspective=True, fovy_or_sizey=None):
        """
        Creates a new single virtual camera.

        Parameters
        ----------
        position : sequence of float, optional
            The position of the camera as 3-element array of xyz coordinates.
        rotation : Rotation, optional
            A rotation from the initial camera orientation (x is right, y is up, z is forward) to the desired one.
        perspective : bool
            Indicates whether this camera uses orthographic (false) or perspective (true) projection.
        fovy_or_sizey : float, optional
            Height of orthographically projected box respectively vertical field of view.
        """

        self.position = np.zeros(3) if position is None else np.asarray(position)
        self.rotation = Rotation.identity() if rotation is None else rotation
        self.perspective = perspective
        if perspective:
            self.fovy = 0.8 if fovy_or_sizey is None else fovy_or_sizey
        else:
            self.sizey = 10 if fovy_or_sizey is None else fovy_or_sizey

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        position = np.asarray(position, dtype=np.float32)
        if position.shape != (3,):
            raise ValueError("Camera expects a single 3-dimensional position vector.")
        self._position = position

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        if not rotation.single:
            raise ValueError("Camera expects a single rotation.")
        self._rotation = rotation

    def get_pitch_yaw_roll(self, degrees=False):
        """
        Expresses the camera's rotation from the initial orientation (x is right, y is up, z is forward) to the desired
        one in terms of pitch (pan up/down), yaw (pan right/left), and roll angles.

        Parameters
        ----------
        degrees : bool
            Whether to return the three angles in radians (false) or degrees (true).

        Returns
        -------
        pitch : float
            How much the camera pans up (positive) or down (negative).
        yaw : float
            How much the camera pans right (positive) or left (negative).
        roll : float
            How much the camera rolls right (positive) or left (negative).
        """

        roll, pitch, neg_yaw = self.rotation.as_euler("zxy", degrees)
        return pitch, -neg_yaw, roll

    def set_pitch_yaw_roll(self, pitch, yaw, roll, degrees=False):
        """
        Sets the camera's rotation from the initial orientation (x is right, y is up, z is forward) to the desired one
        in terms of pitch (pan up/down), yaw (pan right/left), and roll angles.

        Parameters
        ----------
        pitch : float
            How much the camera pans up (positive) or down (negative).
        yaw : float
            How much the camera pans right (positive) or left (negative).
        roll : float
            How much the camera rolls right (positive) or left (negative).
        degrees : bool
            Whether to return the three angles in radians (false) or degrees (true).
        """

        self.rotation = Rotation.from_euler("zxy", [roll, pitch, -yaw], degrees)
        return self

    def zero_roll(self):
        """Zeros the roll component of the camera's rotation, thereby ensuring that the camera is leveled."""
        self.set_pitch_yaw_roll(*self.get_pitch_yaw_roll()[:2], 0.0)
        return self

    def look_at(self, at, up=None):
        """
        Sets the camera's orientation such that it looks at the given point from its own position.

        Parameters
        ----------
        at : sequence of float
            The point that will be in the center of the screen.
        up : sequence of float, optional
            An arbitrary vector in world coordinates that when projected onto the screen points upwards. Defaults to the
            positive y direction.
        """

        if up is None:
            up = np.array([0.0, 1.0, 0.0])
        else:
            up = np.asarray(up)
            up = up / np.linalg.norm(up)
        z = self.position - at
        z /= np.linalg.norm(z)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.rotation = Rotation.from_matrix(np.column_stack([x, y, z]))
        return self

    def view_matrix(self):
        """
        Constructs a 4x4 matrix that transforms from world space to camera space. It captures the position and rotation.

        Returns
        -------
        view_matrix : np.ndarray
        """

        T = np.eye(4, dtype=np.float32)
        R = np.eye(4, dtype=np.float32)
        T[:3, 3] = -self.position
        R[:3, :3] = self.rotation.inv().as_matrix()
        return R @ T

    def projection_matrix(self, aspect, near, far):
        """
        Constructs a 4x4 matrix that transforms from camera space to normalized device coordinates. It captures the
        projection type and camera intrinsics, and depends on the near/far planes and the window's aspect ratio.

        Parameters
        ----------
        aspect : float
            The width of the window or target surface divided by its height.
        near : float
        far : float

        Returns
        -------
        projection_matrix : np.ndarray
        """

        M = np.zeros((4, 4), dtype=np.float32)
        if self.perspective:
            cot = 1 / math.tan(0.5 * self.fovy)
            M[0, 0] = cot / aspect
            M[1, 1] = cot
            M[2, 2] = -(far + near) / (far - near)
            M[2, 3] = -2 * far * near / (far - near)
            M[3, 2] = -1
        else:
            M[0, 0] = 2 / (self.sizey * aspect)
            M[1, 1] = 2 / self.sizey
            M[2, 2] = -2 / (far - near)
            M[2, 3] = -(far + near) / (far - near)
            M[3, 3] = 1
        return M

    def __repr__(self):
        x, y, z = self.position
        pitch, yaw, roll = self.get_pitch_yaw_roll(degrees=True)
        proj = "perspective" if self.perspective else "orthographic"
        sof = f"fovy={self.fovy:.2f}" if self.perspective else f"sizey={self.sizey:.2f}"
        return f"Camera({x=:.2f}, {y=:.2f}, {z=:.2f}, {pitch=:.2f}, {yaw=:.2f}, {roll=:.2f}, {proj}, {sof})"


class CameraPath:
    """
    An immutable interpolatable path of camera objects that passes through a set number of camera keyframes.

    Attributes
    ----------
    times : np.ndarray
        A 1d array with the time index of each keyframe.
    start : float
        The time index of the first keyframe.
    stop : float
        The time index of the last keyframe.
    cameras : list of Camera
        The camera objects that define each keyframe.
    """

    def __init__(self, times, cameras, *, spline=True):
        """
        Setups a new interpolatable camera path.

        Parameters
        ----------
        times : sequence of float
            A sequence with the time index of each keyframe.
        cameras : sequence of Camera
            The camera objects that define each keyframe.
        spline : bool
            Whether to interpolate linearly (false) or using cubic splines (true).
        """

        times = np.asarray(times)
        cameras = list(cameras)
        if times.ndim != 1:
            raise ValueError("Camera path times must be a 1d array.")
        if len(times) != len(cameras):
            raise ValueError("Number of camera path times points must match number of camera objects.")
        if len(times) < 2:
            raise ValueError("Camera path needs at least two keyframes.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Camera path keyframe times must be strictly monotonically increasing.")
        if any(cameras[0].perspective != c.perspective for c in cameras):
            raise ValueError("Camera path camera objects must all share the same projection.")
        self.times = times
        self.cameras = cameras
        positions = np.stack([c.position for c in cameras])
        rotations = Rotation.concatenate([c.rotation for c in cameras])
        fovys_or_sizeys = np.array([c.fovy if c.perspective else c.sizey for c in cameras])
        if not spline:
            self._position_path = lambda t: np.hstack([np.interp(t, times, ps) for ps in positions.T])
            self._rotation_path = Slerp(times, rotations)
            self._fovy_or_sizey_path = lambda t: np.interp(t, times, fovys_or_sizeys)
        else:
            self._position_path = CubicSpline(times, positions)
            self._rotation_path = RotationSpline(times, rotations)
            self._fovy_or_sizey_path = CubicSpline(times, fovys_or_sizeys)

    @property
    def start(self):
        return self.times[0]

    @property
    def stop(self):
        return self.times[-1]

    def __call__(self, t):
        """
        Interpolates the path's camera(s) at the given time index or indices.

        Parameters
        ----------
        t : float or sequence of float
            A scalar time index or a sequence of time indices whose camera object(s) should be interpolated.

        Returns
        -------
        cameras : Camera or list of Camera
            A single interpolated camera if a scalar time index was passed, or a list of interpolated cameras if a
            sequence of time indices was passed.
        """

        t = np.asarray(t)
        if t.ndim > 1:
            raise ValueError("Camera path interpolation time must be either a scalar or a 1d array.")
        if np.any((t < self.start) | (t > self.stop)):
            raise ValueError(f"Time {t} exceeds the camera path's temporal bounds [{self.start}, {self.stop}].")
        perspective = self.cameras[0].perspective
        positions = self._position_path(t)
        rotations = self._rotation_path(t)
        sizeys_or_fovys = self._fovy_or_sizey_path(t)
        if t.ndim == 0:
            return Camera(positions, rotations, perspective, sizeys_or_fovys)
        else:
            return [Camera(p, r, perspective, sof) for p, r, sof in zip(positions, rotations, sizeys_or_fovys)]


class Trackball:
    """
    This class provides facilities to move virtual trackball and extract corresponding camera objects.

    Attributes
    ----------
    anchor : np.ndarray
        The center point around which the camera spins when the trackball is moved.
    distance : float
        The distance between the anchor and the camera.
    camera : Camera
        A readily usable camera object that captures the current state of the trackball.
    """

    def __init__(self, anchor=None, distance=5.0, camera=None):
        """
        Creates a new virtual trackball.

        Parameters
        ----------
        anchor : sequence of float, optional
            The initial center point around which the camera spins when the trackball is moved.
        distance : float
            The initial distance between the anchor and the camera.
        camera : Camera, optional
            This camera object provides the initial rotation, but its initial position is ignored. As the trackball is
            moved, both its position and rotation will continually be modified, and it will directly be exposed to
            clients via the `camera` attribute, meaning that it also defines the camera intrinsics.
        """

        self.anchor = np.zeros(3) if anchor is None else anchor
        self.distance = distance
        self.camera = Camera() if camera is None else camera
        self._compute_camera_position()

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, anchor):
        anchor = np.asarray(anchor)
        if anchor.shape != (3,):
            raise ValueError("Trackball expects a single 3-dimensional anchor vector.")
        self._anchor = anchor

    def rotate(self, dx, dy, speed):
        """
        Rotates the camera around the anchor assuming that the trackball has been moved by the given distance.

        Parameters
        ----------
        dx : float
            The horizontal movement of the trackball to the right (positive) or left (negative).
        dy : float
            The vertical movement of the trackball up (positive) or down (negative).
        speed : float
        """

        self.camera.rotation = self.camera.rotation * Rotation.from_rotvec(np.array([dy, -dx, 0.0]) * speed)
        self._compute_camera_position()

    def translate_on_screen_plane(self, dx, dy, speed):
        """
        Moves the anchor parallel to the screen plane assuming that the trackball has been moved by the given distance.

        Parameters
        ----------
        dx : float
            The horizontal movement of the trackball to the right (positive) or left (negative).
        dy : float
            The vertical movement of the trackball up (positive) or down (negative).
        speed : float
        """

        self.anchor += self.camera.rotation.apply(np.array([-dx, -dy, 0.0])) * self.distance * speed
        self._compute_camera_position()

    def translate_into_screen_plane(self, d, speed):
        """
        Exponentially increases or decreases the distance between the anchor and the camera.

        Parameters
        ----------
        d : float
            The number of steps inward (positive) or outward (negative); may be fractional.
        speed : float
        """

        self.distance = max(0.001, self.distance * (1.0 + speed) ** -d)
        self._compute_camera_position()

    def _compute_camera_position(self):
        self.camera.position = self.anchor + self.camera.rotation.apply(np.array([0.0, 0.0, self.distance]))


from scipy.spatial import KDTree

class AttributedList:

    def __init__(self, primary_name, primary_value, attributes):
        super().__setattr__("_primary_name", primary_name)
        super().__setattr__("_attributes", {})
        self[primary_name] = primary_value
        for name, value in attributes.items():
            self[name] = value

    def __len__(self):
        return self[self._primary_name].shape[0]

    def __getattr__(self, name):
        try:
            return self._attributes[name]
        except KeyError:
            raise AttributeError(f"This {type(self).__name__} object does not have a '{name}' attribute.") from None

    def __setattr__(self, name, value):
        if name == self._primary_name:
            if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != 3:
                c = type(self).__name__
                raise ValueError(f"{c} '{self._primary_name}' must be a numpy array of shape (n_{c.lower()}, 3).")
            old = getattr(self, self._primary_name, None)
            if old is not None and value.shape[0] != old.shape[0]:
                c = type(self).__name__
                raise ValueError(f"Cannot change the number of {c.lower()} in an existing {c} object. "
                                 f"Create a new {c} object instead.")
            super().__setattr__(self._primary_name, np.asarray(value))
        elif value is None:
            if name in self._attributes:
                del self._attributes[name]
        else:
            if not isinstance(value, np.ndarray) or value.ndim == 0 or value.shape[0] != len(self):
                c = type(self).__name__
                raise ValueError(f"{c} attributes must be a numpy array of shape (n_{c.lower()}, *).")
            self._attributes[name] = np.asarray(value)

    def __delattr__(self, name):
        if name == self._primary_name:
            raise AttributeError(f"Cannot delete the '{self._primary_name}' from a {type(self).__name__} object.")
        try:
            del self._attributes[name]
        except KeyError:
            raise AttributeError(f"This {type(self).__name__} object does not have a '{name}' attribute.") from None

    def __getitem__(self, name):
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(name) from None

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __delitem__(self, name):
        try:
            delattr(self, name)
        except AttributeError:
            raise KeyError(name) from None

    def __repr__(self):
        c = type(self).__name__
        elems = []
        for name, value in [(self._primary_name, self[self._primary_name])] + list(self._attributes.items()):
            elems.append(f"{name}=<{'x'.join(map(str, value.shape))} {value.dtype}>")
        return f"{c}({', '.join(elems)})"


class Points(AttributedList):
    """
    A 3D point cloud with optional per-point attributes. To get/set/del an attribute, use dot or indexing notation:

    >>> points = Points(...)
    >>> points.color = np.array([[0.3, 0.5, 0.1], [0.2, 0.9, 0.0]])
    >>> print(points["color"])

    Attributes
    ----------
    position : np.ndarray
        The point positions of shape (n_points, 3).
    attributes : dict
        A dict mapping all per-point attribute names to numpy arrays of shape (n_points, *).
    """

    position: np.ndarray

    @overload
    def __init__(self, points, **attributes):
        """
        Creates a new 3D point cloud from an Open3D PointCloud or trimesh PointCloud object.

        - From an Open3D PointCloud, the colors, covariances, and normals are carried over.
        - From a trimesh PointCloud, the colors are carried over.

        The underlying numpy arrays will be shared.

        Parameters
        ----------
        points : open3d.geometry.PointCloud or trimesh.PointCloud
        **attributes : dict of np.ndarray
            Optional additional per-point attributes of shape (n_points, *).
        """

    @overload
    def __init__(self, position, **attributes):
        """
        Creates a new 3D point cloud with the given positions and optionally a set of per-point attributes.

        Parameters
        ----------
        position : np.ndarray
            The point positions of shape (n_points, 3).
        **attributes : dict of np.ndarray
            The optional per-point attributes of shape (n_points, *).
        """

    def __init__(self, arg1, **attributes):
        if open3d is not None and isinstance(arg1, open3d.geometry.PointCloud):
            points = np.asarray(arg1.points)
            colors = np.asarray(arg1.colors)
            covariances = np.asarray(arg1.covariances)
            normals = np.asarray(arg1.normals)
            if len(colors) == len(points):
                attributes.setdefault("color", np.asarray(colors))
            if len(covariances) == len(points):
                attributes.setdefault("covariance", np.asarray(covariances))
            if len(normals) == len(points):
                attributes.setdefault("normal", np.asarray(normals))
            super().__init__("position", points, attributes)
        elif trimesh is not None and isinstance(arg1, trimesh.PointCloud):
            vertices = arg1.vertices
            colors = arg1.colors
            if len(colors) == len(vertices):
                attributes.setdefault("color", colors)
            super().__init__("position", vertices, attributes)
        else:
            super().__init__("position", arg1, attributes)

    def open3d(self):
        """
        Creates a new Open3D PointCloud including the colors, covariances, and normals.

        The underlying numpy arrays will be shared.

        Returns
        -------
        point_cloud : open3d.geometry.PointCloud
        """

        point_cloud = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(self.position))
        colors = getattr(self, "color", None)
        covariances = getattr(self, "covariance", None)
        normals = getattr(self, "normal", None)
        if colors is not None and colors.shape[1:] == (3,):
            point_cloud.colors = open3d.utility.Vector3dVector(colors)
        if covariances is not None and covariances.shape[1:] == (3,):
            point_cloud.covariances = open3d.utility.Matrix3dVector(covariances)
        if normals is not None and normals.shape[1:] == (3,):
            point_cloud.normals = open3d.utility.Vector3dVector(normals)
        return point_cloud

    def trimesh(self):
        """
        Creates a new trimesh PointCloud including the colors.

        The underlying numpy arrays will be shared.

        Returns
        -------
        point_cloud : trimesh.PointCloud
        """

        point_cloud = trimesh.PointCloud(self.position)
        colors = getattr(self, "color", None)
        if colors is not None and colors.shape[1:] in [(3,), (4,)]:
            point_cloud.colors = colors
        return point_cloud

    @property
    def attributes(self):
        return dict(self._attributes)

    def fit_into_cuboid(self, corner1=-0.9, corner2=0.9):
        """
        Moves and scales the point positions such that they all fit into the cuboid spanned by the supplied corners. All
        axes are scaled by the same factor, and thus the point cloud is not distorted.

        Parameters
        ----------
        corner1, corner2 : float or sequence of float
            Scalar or list/array of length 3 with the xyz coordinates of one cuboid corner.
        """

        pos = self.position
        corners = np.stack(np.broadcast_arrays(corner1, corner2))
        old_bounds_min = np.min(pos, axis=0)
        old_bounds_max = np.max(pos, axis=0)
        new_bounds_min = np.min(corners, axis=0)
        new_bounds_max = np.max(corners, axis=0)
        pos -= 0.5 * (old_bounds_min + old_bounds_max)
        pos *= np.min((new_bounds_max - new_bounds_min) / (old_bounds_max - old_bounds_min))
        pos += 0.5 * (new_bounds_min + new_bounds_max)
        return self

    def wrap_into_cuboid(self, corner1, corner2):
        """
        Wraps around the point positions such that they lie inside the cuboid spanned by the supplied corners. Points
        may lie on the corners themselves. If necessary, points are wrapped multiple times until they fit.

        Parameters
        ----------
        corner1, corner2 : float or sequence of float
            Scalar or list/array of length 3 with the xyz coordinates of one cuboid corner.
        """

        corners = np.stack(np.broadcast_arrays(corner1, corner2))
        corner_min = np.min(corners, axis=0)
        corner_max = np.max(corners, axis=0)
        if corners.ndim == 1:
            self._wrap(self.position, corner_min, corner_max)
        else:
            for col in range(3):
                self._wrap(self.position[:, col], corner_min[col], corner_max[col])
        return self

    @staticmethod
    def _wrap(pos, corner_min, corner_max):
        dim = corner_max - corner_min
        ind = pos < corner_min
        pos[ind] += np.ceil((corner_min - pos[ind]) / dim) * dim
        ind = pos > corner_max
        pos[ind] -= np.ceil((pos[ind] - corner_max) / dim) * dim

    def chamfer_distance(self, other):
        """
        Computes the Chamfer distance between this point cloud and the given point cloud. It is defined as the sum of
        the distances from each point in this cloud to its nearest neighbor in the other cloud, plus the sum of the
        distances from each point in the other cloud to its nearest neighbor in this cloud, divided by the number of
        points in both this and the other cloud.

        Parameters
        ----------
        other : Points

        Returns
        -------
        d : float
        """

        p1 = self.position
        p2 = other.position
        return 0.5 * (np.mean(KDTree(p1).query(p2)[0]) + np.mean(KDTree(p2).query(p1)[0]))


class Faces(AttributedList):
    """
    A list of faces with optional per-face attributes. A face is represented by 3 vertex indices. To get/set/del an
    attribute, use dot or indexing notation:

    >>> faces = Faces(...)
    >>> faces.color = np.array([[0.3, 0.5, 0.1], [0.2, 0.9, 0.0]])
    >>> print(faces["color"])

    Attributes
    ----------
    indices : np.ndarray
        The vertex indices of shape (n_faces, 3).
    attributes : dict
        A dict mapping all per-face attribute names to numpy arrays of shape (n_faces, *).
    """

    indices: np.ndarray

    def __init__(self, indices, **attributes):
        """
        Creates a face list with the given vertex indices and optionally a set of per-face attributes.

        Parameters
        ----------
        indices : np.ndarray
            The vertex indices of shape (n_faces, 3).
        **attributes : dict of np.ndarray
            The optional per-face attributes of shape (n_faces, *).
        """

        super().__init__("indices", indices, attributes)

    @property
    def attributes(self):
        return dict(self._attributes)



class Mesh:
    """
    A 3D mesh whose vertices and faces may both carry arbitrary attributes.

    Attributes
    ----------
    vertices : Points
    faces : Faces
    """

    @overload
    def __init__(self, mesh):
        """
        Creates a new 3D mesh from an Open3D TriangleMesh or Trimesh object.

        - From an Open3D TriangleMesh, the vertex & face normals and vertex colors are carried over, and the
          face-specific vertex uv coordinates are collapsed to vertex uv coordinates.
        - From a Trimesh, the vertex & face normals, vertex & face colors, vertex uv coordinates, and custom vertex and
          face attributes are carried over.

        The underlying numpy arrays will be shared.

        Parameters
        ----------
        mesh : open3d.geometry.TriangleMesh or trimesh.Trimesh
        """

    @overload
    def __init__(self, vertices, faces):
        """
        Creates a new 3D mesh from a set of vertices and faces.

        Parameters
        ----------
        vertices : Points or np.ndarray
            A Points object, or an array with vertex positions of shape (n_points, 3).
        faces : Faces or np.ndarray
            A face list or array of shape (n_faces, 3) that indexes into the vertex list.
        """

    def __init__(self, arg1, arg2=None):
        if open3d is not None and isinstance(arg1, open3d.geometry.TriangleMesh):
            self.vertices = Points(np.asarray(arg1.vertices))
            self.faces = Faces(np.asarray(arg1.triangles))
            vertex_normals = np.asarray(arg1.vertex_normals)
            vertex_colors = np.asarray(arg1.vertex_colors)
            triangle_normals = np.asarray(arg1.triangle_normals)
            triangle_uvs = np.asarray(arg1.triangle_uvs)
            if len(vertex_normals) == len(self.vertices):
                self.vertices.normal = vertex_normals
            if len(vertex_colors) == len(self.vertices):
                self.vertices.color = vertex_colors
            if len(triangle_normals) == len(self.faces):
                self.faces.normal = triangle_normals
            if len(triangle_uvs) == 3 * len(self.faces):
                vertex_uv = np.zeros((len(self.vertices), 2), dtype=triangle_uvs.dtype)
                used_vertex_indices, fst_pos_in_face_arr = np.unique(self.faces.indices.reshape(-1), return_index=True)
                vertex_uv[used_vertex_indices] = triangle_uvs[fst_pos_in_face_arr]
                self.vertices.uv = vertex_uv
        elif trimesh is not None and isinstance(arg1, trimesh.Trimesh):
            self.vertices = Points(arg1.vertices)
            self.faces = Faces(arg1.faces)
            vertex_normals = arg1.vertex_normals
            face_normals = arg1.face_normals
            if len(vertex_normals) == len(self.vertices):
                self.vertices.normal = vertex_normals
            if len(face_normals) == len(self.faces):
                self.faces.normal = face_normals
            visual = arg1.visual
            if isinstance(visual, ColorVisuals):
                vertex_colors = arg1.visual.vertex_colors
                face_colors = arg1.visual.face_colors
                if len(vertex_colors) == len(self.vertices):
                    self.vertices.color = vertex_colors
                if len(face_colors) == len(self.faces):
                    self.faces.color = face_colors
            elif isinstance(visual, TextureVisuals):
                uv = visual.uv
                if uv is not None and len(uv) == len(self.vertices):
                    self.vertices.uv = uv
            for name, value in arg1.vertex_attributes:
                if isinstance(value, np.ndarray) and len(value) == len(self.vertices):
                    self.vertices[name] = value
            for name, value in arg1.face_attributes:
                if isinstance(value, np.ndarray) and len(value) == len(self.faces):
                    self.faces[name] = value
        else:
            self.vertices = arg1 if isinstance(arg1, Points) else Points(arg1)
            self.faces = arg2 if isinstance(arg2, Faces) else Faces(arg2)

    def open3d(self):
        """
        Creates a new Open3D TriangleMesh including the vertex & face normals, vertex colors, and the vertex uv
        coordinates exploded to face-specific vertex uv coordinates.

        The underlying numpy arrays will be shared.

        Returns
        -------
        mesh : open3d.geometry.TriangleMesh
        """

        omesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(self.vertices.position),
            open3d.utility.Vector3iVector(self.faces.indices)
        )
        vertex_normals = getattr(self.vertices, "normal", None)
        vertex_colors = getattr(self.vertices, "color", None)
        face_normals = getattr(self.faces, "normal", None)
        if vertex_normals is not None and vertex_normals.shape[1:] == (3,):
            omesh.vertex_normals = open3d.utility.Vector3dVector(vertex_normals)
        if vertex_colors is not None and vertex_colors.shape[1:] == (3,):
            omesh.vertex_colors = open3d.utility.Vector3dVector(vertex_colors)
        if face_normals is not None and face_normals.shape[1:] == (3,):
            omesh.triangle_normals = open3d.utility.Vector3dVector(face_normals)
        vertex_uv = getattr(self.vertices, "uv", None)
        if vertex_uv is not None and vertex_uv.shape[1:] == (2,):
            omesh.triangle_uvs = open3d.utility.Vector2dVector(vertex_uv[self.faces.indices.reshape(-1)])
        return omesh

    def trimesh(self):
        """
        Creates a new Trimesh including the vertex & face normals, vertex & face colors, vertex uv coordinates, and
        custom vertex and face attributes.

        The underlying numpy arrays will be shared.

        Returns
        -------
        mesh : trimesh.Trimesh
        """

        tmesh = trimesh.Trimesh(self.vertices.position, self.faces.indices, process=False)
        vertex_normals = getattr(self.vertices, "normal", None)
        face_normals = getattr(self.faces, "normal", None)
        if vertex_normals is not None and vertex_normals.shape[1:] == (3,):
            tmesh.vertex_normals = vertex_normals
        if face_normals is not None and face_normals.shape[1:] == (3,):
            tmesh.face_normals = face_normals
        vertex_colors = getattr(self.vertices, "color", None)
        face_colors = getattr(self.faces, "color", None)
        vertex_uv = getattr(self.vertices, "uv", None)
        if (vertex_colors is not None or face_colors is not None) and \
                (vertex_colors is None or vertex_colors.shape[1:] in [(3,), (4,)]) and \
                (face_colors is None or face_colors.shape[1:] in [(3,), (4,)]):
            tmesh.visual = ColorVisuals(tmesh, face_colors, vertex_colors)
        elif vertex_uv is not None and vertex_uv.shape[1:] == (2,):
            tmesh.visual = TextureVisuals(vertex_uv)
        tmesh.vertex_attributes = \
            {n: v for n, v in self.vertices.attributes.items() if n != "normal" and n != "color" and n != "uv"}
        tmesh.face_attributes = \
            {n: v for n, v in self.faces.attributes.items() if n != "normal" and n != "color"}
        return tmesh

    def __repr__(self):
        return f"Mesh({self.vertices}, {self.faces})"



def load_points(path):
    """
    Loads a 3D point cloud from disk.

    Parameters
    ----------
    path : str or file object

    Returns
    -------
    points : Points
    """

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Path to point cloud does not exist or is not a file: {path}")
    if open3d is not None:
        points = Points(open3d.io.read_point_cloud(str(path)))
        if len(points) != 0:
            return points
    if trimesh is not None:
        tmesh = trimesh.load_mesh(path)
        if isinstance(tmesh, trimesh.Scene):
            tmesh = tmesh.dump(concatenate=True)
        return Mesh(tmesh.vertices)
    raise ValueError(f"Neither Open3D nor trimesh (if installed) could load the point cloud: {path}")


def save_points(path, points):
    """
    Saves a 3D point cloud to disk.

    Parameters
    ----------
    path : str or file object
    points : Points
    """

    path = Path(path)
    if open3d is not None:
        open3d.io.write_point_cloud(str(path), points.open3d())
        return
    if trimesh is not None:
        points.trimesh().export(path)
        return
    raise ValueError(f"Neither Open3D nor trimesh (if installed) could save the point cloud to: {path}")


def load_mesh(path):
    """
    Loads a 3D mesh from disk.

    Parameters
    ----------
    path : str or file object

    Returns
    -------
    mesh : Mesh
    """

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Path to mesh does not exist or is not a file: {path}")
    if open3d is not None:
        mesh = Mesh(open3d.io.read_triangle_mesh(str(path)))
        if len(mesh.vertices) != 0:
            return mesh
    if trimesh is not None:
        tmesh = trimesh.load_mesh(path)
        if isinstance(tmesh, trimesh.Scene):
            tmesh = tmesh.dump(concatenate=True)
        return Mesh(tmesh)
    raise ValueError(f"Neither Open3D nor trimesh (if installed) could load the mesh: {path}")


def save_mesh(path, mesh):
    """
    Saves a 3D mesh to disk.

    Parameters
    ----------
    path : str or file object
    mesh : Mesh
    """

    path = Path(path)
    if open3d is not None:
        open3d.io.write_triangle_mesh(str(path), mesh.open3d())
        return
    if trimesh is not None:
        mesh.trimesh().export(path)
        return
    raise ValueError(f"Neither Open3D nor trimesh (if installed) could save the mesh to: {path}")


ldr_extensions = [".jpg", ".png"]
hdr_extensions = [".exr", ".hdr"]

def is_hdr_from_file_extension(file_path):
    extension = os.path.splitext(file_path)[1]
    return extension in hdr_extensions

def load_image(path, normalize=True, append_alpha=False):
    
    assert os.path.isfile(path), "Image file does not exist"
    is_hdr = is_hdr_from_file_extension(path)
    flags = (cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) if is_hdr else cv2.IMREAD_UNCHANGED

    img = cv2.imread(path, flags)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize and not is_hdr:
        img = img.astype(np.float32) / 255.
    if append_alpha and img.shape[2] == 3:
        alpha = np.ones_like(img[..., 0:1])
        img = np.concatenate([img, alpha], axis=-1)
    print(img.shape, img.min(), img.max())
    return img


class OpenGLOP:

    def __init__(
        self,
        vertex_shaders, 
        geometry_shaders,
        fragment_shaders, 
        rendertarget_count, 
        rendertarget_resolution=None,
        create_depth_rendertargets=False,
        defines=dict()):
        
        # init shader(s)
        self.shaders = []
        if not isinstance(vertex_shaders, list):
            vertex_shaders = [vertex_shaders]
        if not isinstance(geometry_shaders, list):
            geometry_shaders = [geometry_shaders]
        if not isinstance(fragment_shaders, list):
            fragment_shaders = [fragment_shaders]
        assert len(vertex_shaders) == len(geometry_shaders) == len(fragment_shaders), "Shader list lengths must match."
        for v, g, f in zip(vertex_shaders, geometry_shaders, fragment_shaders):
            self.shaders.append(create_shader(v, g, f, defines))
        if len(self.shaders) == 1:
            self.shader = self.shaders[0]

        # init uniforms
        self.uniforms = {}
        
        # init render target(s)
        self.rendertarget_count = rendertarget_count
        if rendertarget_count > 0:
            assert rendertarget_resolution is not None, "Need resolution to create render target."
            if isinstance(rendertarget_resolution, int):
                rendertarget_resolution = (rendertarget_resolution, rendertarget_resolution)
            self.rendertargets = []
            for _ in range(rendertarget_count):
                self.rendertargets.append(self.create_rendertargets(rendertarget_resolution, create_depth_rendertargets))

    #----------------------------

    # initialize uniforms of a shader
    def init_uniform(self, name, shader_idx=0):
        self.uniforms[name] = glGetUniformLocation(self.shaders[shader_idx], name)

    #----------------------------

    # create render targets for rendering to a texture
    def create_rendertargets(self, res, create_depthbuffer, channels=4):        
        rt = Rendertarget()
        rt.color = Texture2D()
        rt.color.allocate_memory(res, channels)
        if create_depthbuffer:
            rt.depth = Texture2D()
            rt.depth.allocate_memory(res, depth_texture=True)
        return rt
    

from abc import ABC, abstractmethod
# an abstract class for an OpenGL texture
class Texture(ABC):

    @abstractmethod
    def __init__(self):
        self.handle = glGenTextures(1)
        self._resolution = (0, 0)
        self._channels = 0
        self._gl_format = None
        self.target = None
        self.need_allocation = True

    # def __del__(self):
    #     glDeleteTextures(1, [self.handle])

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        if self._resolution != value:
            self.need_allocation = True
        self._resolution = value

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, value):
        if self._channels != value:
            self.need_allocation = True
        self._channels = value

    @property
    def gl_format(self):
        return self._gl_format

    @gl_format.setter
    def gl_format(self, value):
        if self._gl_format != value:
            self.need_allocation = True
        self._gl_format = value

    @property
    def max_mip_levels(self):
        return int(math.log(max(self.resolution[0], self.resolution[1]), 2)) + 1

    # bind the texture
    def bind(self):
        glBindTexture(self.target, self.handle)

    # unbind the texture
    def unbind(self):
        glBindTexture(self.target, 0)

    # set sampling parameters
    def set_params(self, min_filter=GL_NEAREST, mag_filter=GL_LINEAR, wrap=GL_CLAMP_TO_BORDER):
        self.bind()
        glTexParameteri(self.target, GL_TEXTURE_MIN_FILTER, min_filter)
        glTexParameteri(self.target, GL_TEXTURE_MAG_FILTER, mag_filter)
        glTexParameterfv(self.target, GL_TEXTURE_BORDER_COLOR, [0, 0, 0, 1])
        glTexParameteri(self.target, GL_TEXTURE_WRAP_S, wrap)
        glTexParameteri(self.target, GL_TEXTURE_WRAP_T, wrap)

    # generate a linear MIP map
    def build_MIP(self):
        self.bind()
        glGenerateMipmap(self.target)

    # infer format and internal format from channel count
    @staticmethod
    def gl_format_from_channel_count(c):
        assert c > 0 and c < 5, "Channel count can only be in [1-4]"
        formats = [
            (GL_R32F, GL_RED),
            (GL_RG32F, GL_RG),
            (GL_RGB32F, GL_RGB),
            (GL_RGBA32F, GL_RGBA)
        ]
        return formats[c-1]


#=====================================================================

# an OpenGL 2D texture
class Texture2D(Texture):

    def __init__(self, image=None, flip_h=True):
        super().__init__()
        self.target = GL_TEXTURE_2D
        if image is not None:
            self.upload_image(image, flip_h=flip_h)

    # allocate GPU memory for the texture
    def allocate_memory(self, resolution, channels=4, depth_texture=False, force_allocation=False):
        self.resolution = resolution
        if depth_texture:
            self.channels = 1
            self.gl_format = GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT
        else:
            self.channels = channels
            self.gl_format = self.gl_format_from_channel_count(channels)
        if self.need_allocation or force_allocation:
            self.bind()
            glTexImage2D(self.target, 0, self.gl_format[0], resolution[0], resolution[1], 0, self.gl_format[1], GL_FLOAT, None)
            self.need_allocation = False

    # upload an image into the texture
    def upload_image(self, image, flip_h=True):
        if image.ndim == 2:
            image = image[..., None]
        w, h, self.channels = image.shape
        self.resolution = (w, h)
        if flip_h:
            image = np.flip(image, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        self.gl_format = self.gl_format_from_channel_count(self.channels)
        self.bind()
        glTexImage2D(self.target, 0, self.gl_format[0], self.resolution[1], self.resolution[0], 0, self.gl_format[1], GL_FLOAT, image)
        self.need_allocation = False

    # download the texture as an image
    def download_image(self, flip_h=True):
        self.bind()

        # PyOpenGL does not support downloading GL_RG textures
        # therefore: make it GL_RGB temporarily
        temp_channels = 3 if self.channels == 2 else self.channels

        _, gl_format = self.gl_format_from_channel_count(temp_channels)
        image = glGetTexImage(self.target, 0, gl_format, GL_FLOAT)
        self.unbind()
        if self.channels == 1:
            image = image[..., None]
        w, h, c = image.shape
        image = np.reshape(image, (h, w, c))

        # if temp channel was necessary, throw it away
        if not temp_channels == self.channels:
            image = image[..., 0:2]

        if flip_h:
            image = np.flip(image, 0)
        return image




class VertexArrayObject:
    """
    An OpenGL vertex array object (VAO) that bundles of a set of vertices, each of which may have arbitrary attributes,
    with an optional vertex index array that defines the order of the vertices and permits a vertex to occur more than
    once in the vertex stream.
    """

    def __init__(self):
        self._handle = glGenVertexArrays(1)
        self._attribute_buffer_handles = {}
        self._attribute_vertex_count = None
        self._index_buffer_handle = None
        self._index_buffer_count = None
        self._index_buffer_gl_type = None

    def bind(self):
        """Binds this vertex array object."""
        glBindVertexArray(self._handle)

    def unbind(self):
        """Unbinds the currently bound vertex array object."""
        glBindVertexArray(0)

    def upload_points(self, points, shader):
        """
        Uploads all attributes (including `position`) of the given points object that also appear in the given shader
        to the VAO, and binds them to the corresponding shader inputs.

        Parameters
        ----------
        points : Points
        shader
            The shader to which the attributes will be bound.
        """
        self.upload_attribute(points.position, "position", shader)
        for name, value in points.attributes.items():
            if glGetAttribLocation(shader, name) >= 0 and value.ndim == 2 and 1 <= value.shape[1] <= 4 and \
                    (np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)):
                self.upload_attribute(value, name, shader)
        return self

    def upload_mesh(self, mesh, shader):
        """
        Uploads all vertex attributes (including `position`) of the given mesh that also appear in the given shader to
        the VAO, binds them to the corresponding shader inputs, and then also uploads the face `indices`.

        Parameters
        ----------
        mesh : Mesh
        shader
            The shader to which the vertex attributes will be bound.
        """
        self.upload_points(mesh.vertices, shader)
        self.upload_indices(mesh.faces.indices)
        return self

    def upload_attribute(self, array, name, shader):
        """
        Uploads the given vertex attribute to the VAO and binds it to the shader input with the given name.

        Parameters
        ----------
        array : np.ndarray
            The vertex attribute of shape (n_vertices, 1/2/3/4).
        name : str
            The name of the shader input to which the vertex attribute will be bound.
        shader
            The shader to which the vertex attribute will be bound.
        """

        if array.ndim != 2 or not 1 <= array.shape[1] <= 4:
            raise ValueError("Vertex attribute array must have shape (n_vertices, 1/2/3/4).")
        dtype = array.dtype
        if dtype == np.int8:
            gl_type = GL_BYTE
        elif dtype == np.int16:
            gl_type = GL_SHORT
        elif dtype == np.int32:
            gl_type = GL_INT
        elif dtype == np.uint8:
            gl_type = GL_UNSIGNED_BYTE
        elif dtype == np.uint16:
            gl_type = GL_UNSIGNED_SHORT
        elif dtype == np.uint32:
            gl_type = GL_UNSIGNED_INT
        elif dtype == np.float16:
            gl_type = GL_HALF_FLOAT
        elif dtype == np.float32:
            gl_type = GL_FLOAT
        elif np.issubdtype(dtype, np.signedinteger):
            gl_type = GL_INT
            array = array.astype(np.int32)
        elif np.issubdtype(dtype, np.unsignedinteger):
            gl_type = GL_UNSIGNED_INT
            array = array.astype(np.uint32)
        elif np.issubdtype(dtype, np.floating):
            gl_type = GL_FLOAT
            array = array.astype(np.float32)
        else:
            raise ValueError(f"dtype of vertex attribute array not recognized: {dtype}")
        print("get location", shader, name)
        attr_loc = glGetAttribLocation(shader, name)
        if attr_loc < 0:
            raise ValueError(f"The passed shader does not have a '{name}' attribute.")
        if self._attribute_vertex_count is None:
            self._attribute_vertex_count = array.shape[0]
        elif self._attribute_vertex_count != array.shape[0]:
            raise ValueError(f"Attributes with different vertex counts uploaded to VAO: {self._attribute_vertex_count} "
                             f"and {array.shape[0]}.")
        if name not in self._attribute_buffer_handles:
            self._attribute_buffer_handles[name] = glGenBuffers(1)
        self.bind()
        glBindBuffer(GL_ARRAY_BUFFER, self._attribute_buffer_handles[name])
        glEnableVertexAttribArray(attr_loc)
        glVertexAttribPointer(attr_loc, array.shape[1], gl_type, GL_FALSE, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, array, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.unbind()
        return self

    def upload_indices(self, indices):
        """
        Uploads the vertex index array that defines the order of the vertices and permits a vertex to occur more than
        once in the vertex stream. If the array has more than one dimension (e.g., think about a triangle array of shape
        (n_faces, 3)), it will be flattened.

        Parameters
        ----------
        indices : np.ndarray
            An arbitrarily-dimensional array that indices into the set of vertices.
        """

        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Vertex array indices must be integers.")
        size = indices.dtype.itemsize
        if size == 1:
            self._index_buffer_gl_type = GL_UNSIGNED_BYTE
        elif size == 2:
            self._index_buffer_gl_type = GL_UNSIGNED_SHORT
        else:
            self._index_buffer_gl_type = GL_UNSIGNED_INT
            size = 4
        indices = indices.reshape(-1).astype(f"uint{size * 8}")
        self._index_buffer_count = len(indices)
        if self._index_buffer_handle is None:
            self._index_buffer_handle = glGenBuffers(1)
        self.bind()
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._index_buffer_handle)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)
        self.unbind()
        return self

    def draw(self, mode=GL_TRIANGLES, *, use_indices=True):
        """
        Calls `glDrawElements` or `glDrawArrays` with the VAO being bound.

        Parameters
        ----------
        mode
            The `GL_*` mode that interprets and draws the vertex stream.
        use_indices : bool
            If true, use the vertex index array (if uploaded) to obtain the vertex stream; otherwise, use the vertices
            in the order their attributes have been uploaded.
        """

        self.bind()
        if use_indices and self._index_buffer_count is not None:
            glDrawElements(mode, self._index_buffer_count, self._index_buffer_gl_type, ctypes.c_void_p(0))
        elif self._attribute_vertex_count is not None:
            # print("in the draw function I am")
            glDrawArrays(mode, 0, self._attribute_vertex_count)
        self.unbind()

class Framebuffer:

    def __init__(self):
        self.handle = glGenFramebuffers(1)

    #def __del__(self):
        #glDeleteFramebuffers(1, [self.handle])

    def bind(self, target=GL_FRAMEBUFFER):
        glBindFramebuffer(target, self.handle)

    def unbind(self, target=GL_FRAMEBUFFER):
        glBindFramebuffer(target, 0)

def render_screen_quad():
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(-1, -1)
    glTexCoord2f(0, 1)
    glVertex2f(1, -1)
    glTexCoord2f(1, 1)
    glVertex2f(1, 1)
    glTexCoord2f(1, 0)
    glVertex2f(-1, 1)
    glEnd()

class ImageDisplayOP(OpenGLOP):

    def __init__(self, res, rendertarget_count=0):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            os.path.join(this_dir, "shaders", "vertex2D_uv"),
            None,
            os.path.join(this_dir, "shaders", "textured_quad"), 
            rendertarget_count,
            rendertarget_resolution=res)
        self.init_uniform("outputRes")
        self.init_uniform("showArray")
        self.init_uniform("level")
        self.init_uniform("layer")        
        self.init_uniform("showOverlay")
        self.init_uniform("overlayPosition")
        self.res = res

    def render(self, tex, to_screen=False, overlay_tex=None, overlay_pos=(0, 0), level=0, layer=0, rendertarget_id=0):
    
        if not to_screen:
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
                self.rendertargets[rendertarget_id].color.handle, 0)

        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, self.res[0], self.res[1])
        glUseProgram(self.shader)

        if type(tex) == Texture2D:
            glActiveTexture(GL_TEXTURE0)
            glProgramUniform1i(self.shader, self.uniforms["showArray"], False)
        else:
            glActiveTexture(GL_TEXTURE1)
            glProgramUniform1i(self.shader, self.uniforms["showArray"], True)
        
        tex.bind()
        min_filter = GL_NEAREST if level == 0 else GL_NEAREST_MIPMAP_NEAREST
        tex.set_params(min_filter=min_filter)

        glProgramUniform2i(self.shader, self.uniforms["outputRes"], *self.res)
        glProgramUniform1i(self.shader, self.uniforms["level"], level)
        glProgramUniform1i(self.shader, self.uniforms["layer"], layer)
        
        if overlay_tex is not None:
            glProgramUniform1i(self.shader, self.uniforms["showOverlay"], True)
            glActiveTexture(GL_TEXTURE2)
            overlay_tex.bind()
            overlay_tex.set_params()
            glProgramUniform2i(self.shader, self.uniforms["overlayPosition"], *overlay_pos)
        else:
            glProgramUniform1i(self.shader, self.uniforms["showOverlay"], False)

        glClear(GL_COLOR_BUFFER_BIT)
        render_screen_quad()
        
        tex.unbind()
        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)

        if not to_screen:
            return self.rendertargets[rendertarget_id].color


class TextOP(OpenGLOP):

    def __init__(self, resolution, font_size=50, rendertarget_count=1):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            os.path.join(this_dir, "shaders", "text"),
            None,
            os.path.join(this_dir, "shaders", "text"), 
            rendertarget_count, 
            resolution)
        self.init_uniform("modelMatrix")
        self.init_uniform("textColor")

        self.characters = []

        self.make_font(os.path.join(this_dir,'arial.ttf'), font_size)

    #--------------------

    def make_font(self, filename, font_size):
        
        face = freetype.Face(filename)
        face.set_pixel_sizes(0, font_size)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glActiveTexture(GL_TEXTURE0)

        for c in range(128):
            face.load_char(chr(c), freetype.FT_LOAD_RENDER)
            glyph   = face.glyph
            bitmap  = glyph.bitmap
            size    = bitmap.width, bitmap.rows
            bearing = glyph.bitmap_left, glyph.bitmap_top 
            advance = glyph.advance.x

            # create glyph texture
            tex = Texture2D()
            tex.bind()
            tex.set_params(mag_filter=GL_LINEAR, min_filter=GL_LINEAR)
            glTexImage2D(tex.target, 0, GL_R8, *size, 0, GL_RED, GL_UNSIGNED_BYTE, bitmap.buffer)
            self.characters.append((tex, size, bearing, advance))

        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        tex.unbind()

    #--------------------

    def render(self, text, position=[15, 15], scale=1., color=[1, 1, 1], background_color=[0, 0, 0, 0], rendertarget_id=0):
        assert rendertarget_id < self.rendertarget_count
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
            self.rendertargets[rendertarget_id].color.handle, 0)
        glViewport(0, 0, *self.rendertargets[rendertarget_id].color.resolution)
        
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(self.shader)
                
        global_shift_matrix = translation_matrix_3D((position[0], -position[1], 0))
        proj_matrix = ortho_projection_matrix(
            (self.rendertargets[rendertarget_id].color.resolution[0], 
            -self.rendertargets[rendertarget_id].color.resolution[1]))

        glProgramUniform3f(self.shader, self.uniforms["textColor"], *color)
        
        glClearColor(*background_color)
        glClear(GL_COLOR_BUFFER_BIT)

        glActiveTexture(GL_TEXTURE0)

        char_x = 0
        for c in text:
                    
            c = ord(c)
            ch          = self.characters[c]
            w, h        = ch[1][0] * scale, ch[1][1] * scale
            xrel, yrel  = char_x + ch[2][0] * scale, (ch[1][1] - ch[2][1]) * scale
            char_x     += (ch[3] >> 6) * scale

            scale_matrix = anisotropic_scaling_matrix_3D((w, h, 1))
            rel_shift_matrix = translation_matrix_3D((xrel, yrel, 0))
            model_matrix = proj_matrix @ global_shift_matrix @ rel_shift_matrix @ scale_matrix
            glProgramUniformMatrix4fv(self.shader, self.uniforms["modelMatrix"], 1, True, *model_matrix)
            
            ch[0].bind()    
            render_screen_quad()
            
        glUseProgram(0)
        glDisable(GL_BLEND)

        return self.rendertargets[rendertarget_id].color
    
# create a batch of dim x dim identity matrices for pytorch
def torch_identity_matrix_boilerplate(dim, batch_size, to_cuda=False):
    m = torch.eye(dim, dtype=torch.float32)
    if to_cuda:
        m = m.cuda()
    return m.repeat(batch_size, 1, 1)

# prepare auxiliary matrix parameters for pytorch
def torch_aux_params_boilerplate(x, to_cuda=False):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    batch_size = x.shape[0]
    if to_cuda:
        x = x.cuda()
    return x, batch_size

#-----------------------------------------------
# actual matrix creation

# create a 2D translation matrix
def translation_matrix_2D(translation):
    m = np.eye(3, dtype=np.float32)
    m[0:2, 2] = translation
    return m

# pytorch version of the above, including batch dimension
def translation_matrix_2D_torch(translation, to_cuda=False):
    translation, batch_size = torch_aux_params_boilerplate(translation, to_cuda)
    m = torch_identity_matrix_boilerplate(3, batch_size, to_cuda)
    m[:, :2, 2] = translation
    return m

# create a 3D translation matrix
def translation_matrix_3D(translation):
    m = np.eye(4, dtype=np.float32)
    m[0:3, 3] = translation
    return m

# pytorch version of the above, including batch dimension
def translation_matrix_3D_torch(translation, to_cuda=False):
    translation, batch_size = torch_aux_params_boilerplate(translation, to_cuda)
    m = torch_identity_matrix_boilerplate(4, batch_size, to_cuda)
    m[:, :3, 3] = translation
    return m

# create a 2D isotropic scaling matrix
def isotropic_scaling_matrix_2D(s):
    m = np.eye(3, dtype=np.float32)
    m[0:2, 0:2] *= s
    return m

# pytorch version of the above, including batch dimension
def isotropic_scaling_matrix_2D_torch(s, to_cuda=False):
    s, batch_size = torch_aux_params_boilerplate(s, to_cuda)
    m = torch_identity_matrix_boilerplate(3, batch_size, to_cuda)
    m[:, :2, :2] *= s.view(-1, 1, 1)
    return m

# create a 3D isotropic scaling matrix
def isotropic_scaling_matrix_3D(s):
    m = np.eye(4, dtype=np.float32)
    m[0:3, 0:3] *= s
    return m

# pytorch version of the above, including batch dimension
def isotropic_scaling_matrix_3D_torch(s, to_cuda=False):
    s, batch_size = torch_aux_params_boilerplate(s, to_cuda)
    m = torch_identity_matrix_boilerplate(4, batch_size, to_cuda)
    m[:, :3, :3] *= s.view(-1, 1, 1)
    return m

# create a 2D anisotropic scaling matrix
def anisotropic_scaling_matrix_2D(s):
    m = np.eye(3, dtype=np.float32)
    m[0,0] = s[0]
    m[1,1] = s[1]
    return m

# pytorch version of the above, including batch dimension
def anisotropic_scaling_matrix_2D_torch(s, to_cuda=False):
    s, batch_size = torch_aux_params_boilerplate(s, to_cuda)
    m = torch_identity_matrix_boilerplate(3, batch_size, to_cuda)
    m[:, 0, 0] = s[:, 0]
    m[:, 1, 1] = s[:, 1]
    return m

# create a 3D anisotropic scaling matrix
def anisotropic_scaling_matrix_3D(s):
    m = np.eye(4, dtype=np.float32)
    m[0,0] = s[0]
    m[1,1] = s[1]
    m[2,2] = s[2]
    return m

# pytorch version of the above, including batch dimension
def anisotropic_scaling_matrix_3D_torch(s, to_cuda=False):
    s, batch_size = torch_aux_params_boilerplate(s, to_cuda)
    m = torch_identity_matrix_boilerplate(4, batch_size, to_cuda)
    m[:, 0, 0] = s[:, 0]
    m[:, 1, 1] = s[:, 1]
    m[:, 2, 2] = s[:, 2]
    return m

# map [0, w[0]]*[0, w[1]] to [-1, 1]^2
def ortho_projection_matrix(w):
    s = anisotropic_scaling_matrix_3D((2/w[0], 2/w[1], 1))
    t = translation_matrix_3D((-1, -1, 0))
    return t @ s



# save an image
def save_image(img, path, channels=3, jpeg_quality=95):
    is_hdr = is_hdr_from_file_extension(path)

    if img.ndim == 2:
        out_img = img[..., None]
    if img.ndim == 3 and img.shape[2] >= 2:
        if channels == 2:
            out_img = np.zeros((*img.shape[0:2], 3))
            out_img[..., 1:3] = img[..., 2::-1]
        if channels == 3:
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if channels == 4:
            out_img = cv2.cv2Color(img, cv2.COLOR_RGBA2BGRA)
    if (out_img.dtype == np.float32 or out_img.dtype == np.float64) and not is_hdr:
        out_img = np.clip(out_img, 0, 1) * 255
        out_img = out_img.astype(np.uint8)
    if is_hdr:
        out_img = out_img.astype(np.float32)
        
    cv2.imwrite(path, out_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    
def print_same_line(v):
    print("\r"+str(v), end="")

def now_string():
   return datetime.utcnow().strftime("%Y.%m.%d_%H.%M.%S.%f")[:-3]

class OpenGLWindow:

    def __init__(self, display_res=(1,1), title="OpenGL Window", logging_level=logging.INFO, max_fps = None):
        
        assert OGL_FRAMEWORK == OpenGLFramework.glfw, "Can only use GLFW to create a window"

        self.window_handle = glfw.create_window(display_res[0], display_res[1], title, None, None)
        assert self.window_handle, "Unable to create GLFW window"
        glfw.make_context_current(self.window_handle)    

        self.fbo = Framebuffer()

        glfw.set_key_callback(self.window_handle, self.key_callback)
        glfw.set_mouse_button_callback(self.window_handle, self.mouse_click_callback)
        glfw.set_cursor_pos_callback(self.window_handle, self.mouse_move_callback)
        glfw.set_scroll_callback(self.window_handle, self.mouse_scroll_callback)

        logging.basicConfig(level=logging_level, format='%(levelname)s [%(funcName)s]: %(message)s')

        self.display_res = display_res
        self.max_fps = max_fps
        self.calc_fps = None

        self.mods = 0

        self.mouse_is_clicked = False
        self.mouse_is_moved = False

        self.left_mouse_down = False
        self.right_mouse_down = False
        self.middle_mouse_down = False
        self.click_position = None

        self.display_count = 1
        self.display_idx = 1
        self.display_stats = True
        self.stats_frequency = 0.25

        self.is_recording_frame = False 
        self.is_recording_display = False
        self.frame_recording_buffer = []

        self.frame_count = 0
        self.lazy_update = False
        self.need_update = True
        
        try:
            self.init_shaders()
        except:
            logging.exception("Could not initialize shaders in OpenGLWindow. Manually call init_shaders() in derived class.")

        #self.fbo.bind()

        print("----- Press H for help window. -----")

        # Default keys: # Add more key bindings as needed using the self.register_key method..
        # update this dict if keys are within some range...
        self.key_bindings = {
            "0 & 1-9" : "Set Display 10, 1-9"
            
        }

        # Update help dictionary
        self.default_key_press(None)
        self.key_press(None)


    #----------------------------------

    # initialize all shaders
    def init_shaders(self):
        self.display_op = ImageDisplayOP(self.display_res)
        self.performance_text_op = TextOP((250, 37), font_size=17)

    #----------------------------------

    # starts the main display loop
    def run(self):
        logging.info("Entering main loop.")
        last_performance_update = time.perf_counter() 
        performance_tex = None
        self.render_tex = None
        self.fps_count = 0
        smoothing_factor = 0.9
        warm_up_time = 10
        while not glfw.window_should_close(self.window_handle):
            if self.need_update:
                self.fbo.bind()
                time_before = time.perf_counter() 
                textures = self.render() # the main render function
                if self.display_stats or self.is_recording_display:
                    glFinish()
                    time_after = time.perf_counter()
                    if (time_after - last_performance_update) > self.stats_frequency and self.fps_count > warm_up_time:
                        time_elapsed = (time_after - time_before) * 1000
                        stats_text = (  f"[{self.display_idx}/{self.display_count}]    "
                                        f"{time_elapsed:0.1f} ms ({self.max_fps if self.max_fps != None else self.calc_fps} fps) ")
                        
                        if self.is_recording_display: 
                            stats_text += f"[{self.frame_count}]" 
                        stats_color = [0, 0, 0] if self.is_recording_display else [1, 1, 1]
                        background_color = [0.8, 0, 0, 0.7] if self.is_recording_display else [0, 0, 0, 0.7]
                        performance_tex = self.performance_text_op.render(stats_text, position=[15, 12], color=stats_color, background_color=background_color)
                        last_performance_update = time_after
                else:
                    performance_tex = None
                self.fbo.unbind()

                # ----------------------------------------------------

                if textures is not None:
                    if not isinstance(textures, tuple):
                        textures = (textures,)
                    self.display_count = len(textures)
                    self.display_idx = min(self.display_idx, self.display_count)
                    self.render_tex = textures[self.display_idx-1]
                    if self.is_recording_frame:
                        self.frame_recording_buffer.append(self.render_tex.download_image())
                        self.frame_count += 1

                # ----------------------------------------------------
                
                if self.fps_count > warm_up_time:
                    self.need_update = False

            self.display_op.render(self.render_tex, overlay_tex=performance_tex, to_screen=True)
            glfw.swap_buffers(self.window_handle)
            glfw.poll_events()

            self.post_render()    
            
            # compute_fps
            # ----------------------------------------------------
            if self.need_update:
                time_end = time.perf_counter()
                render_time = time_end - time_before
                if self.max_fps != None:
                    if render_time < (1 / self.max_fps):
                        time.sleep((1 / self.max_fps) - render_time)
                else:
                    self.calc_fps = 1/render_time
                    time.sleep(1e-100)
                
                if not self.fps_count == 0:
                    # Exponential moving average
                    self.calc_fps = (self.calc_fps * smoothing_factor) + (self.prev_fps * (1 - smoothing_factor))
                                
                self.calc_fps = round(self.calc_fps)
                self.prev_fps = self.calc_fps

                self.fps_count += 1
            
            # ----------------------------------------------------
            
        glfw.terminate()

    #----------------------------------
    @property
    def need_update(self):
        return self._need_update
    
    @need_update.setter
    # set flag to compute the render pass
    def need_update(self, value):
        if self.lazy_update:
            # compute frame only when a click and dragging motion is performed..pure mouse movement doesnot affect anything
            if self.mouse_is_moved:
                if self.mouse_is_clicked:
                    self._need_update = value
                self.mouse_is_moved = not self.mouse_is_moved
                return
            
            self._need_update = value
        else:
            self._need_update = True

    #----------------------------------

    # overwrite this function with your custom render code
    def render(self):
        pass
    
    #----------------------------------
    # overwrite this function with your custom post render code
    def post_render(self):
        pass
    
    #----------------------------------

    def save_frames(self):
        root = tk.Tk()
        root.withdraw()
        recording_directory = filedialog.askdirectory()
        if recording_directory:
            for idx, frame in enumerate(self.frame_recording_buffer):
                print_same_line(f"Saving frame {idx+1}/{len(self.frame_recording_buffer)} to disk.")
                save_image(frame, os.path.join(recording_directory, f"recording_{idx:05}.png"))
            print("")
        self.frame_count = 0
        self.frame_recording_buffer = []

    #----------------------------------

    def register_key(self, register_key, match_key=None, description=""):
        if match_key is None:
            self.key_bindings.update({
                register_key: f"{description}"
                })
            return False
        if isinstance(register_key, tuple):
            return register_key[1] == match_key and self.mods == register_key[0]
        else:
            # When no modifiers are present, strictly activate the registered key.
            # This (self.mods == 0) prevents collision with the above condition for keys with modifiers.
            return register_key == match_key and self.mods == 0

            
    #----------------------------------

    # handle key inputs
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.need_update = True
            self.default_key_press(key)
            self.key_press(key)
            
        if action == glfw.RELEASE:

            # register shift and ctrl release
            if key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
                self.mods = 0
            if key == glfw.KEY_LEFT_CONTROL or key == glfw.KEY_RIGHT_CONTROL:
                self.mods = 0

            self.key_release(key)

    #----------------------------------

    # overwrite these functions with custom key events
    def key_press(self, key):
        pass
        
    def default_key_press(self, key):
        def glfw_key(key):
            return key_mappings.get(key, f"{key}")
        
        if self.register_key(glfw.KEY_ESCAPE, key, "Exit main loop."):
            logging.info("Exiting main loop.")
            glfw.set_window_should_close(self.window_handle, True)

        if self.register_key(glfw.KEY_H, key, "Show Help Window."):
            print()
            print("------- Help Window ----------")
            print()
            for k, description in self.key_bindings.items():
                if isinstance(k, tuple):
                    readable_key = "+".join([glfw_key(key) for key in k])
                else:
                    readable_key = glfw_key(k)

                # Beautify layout # 30 coz that's the 2* (maximum string length available)
                if len(readable_key) < 30:
                    readable_key += ' ' * (30 - len(readable_key)) 
                tabs = '\t' * 2

                formatted_string = f"{readable_key}{tabs}{description}"
                print(formatted_string)
            print()
            print("------------------------------")
            print()

        # display stats
        if self.register_key(glfw.KEY_P, key, "Toggle display stats."):
            self.display_stats = not self.display_stats

        # Lazy update display
        if self.register_key(glfw.KEY_T, key, "Toggle lazy update."):
            self.lazy_update = not self.lazy_update
            self.need_update = not self.need_update
            self.fps_count = 0
            logging.info(f"Lazy Update : {self.lazy_update}")

        # reload shaders
        if self.register_key(glfw.KEY_R, key, "Reload shaders."):
            logging.info("Reloading shaders.")
            try:
                self.init_shaders()
            except RuntimeError as error:
                logging.error(error)

        # send current image to tev
        if self.register_key((glfw.MOD_CONTROL, glfw.KEY_S), key, "Send current image to Tev."):
            logging.info("Sending image to Tev.")
            img = self.render_tex.download_image()
            # send_to_tev(now_string(), img)
            
        # start and stop recording
        if self.register_key(glfw.KEY_O, key, "Start/stop recording."):
            if not self.is_recording_frame:
                logging.info("Starting recording.")
            else:
                logging.info("Stopped recording.")
                self.save_frames()
            self.is_recording_frame = not self.is_recording_frame
            self.is_recording_display = not self.is_recording_display

        # select texture for display
        def set_display(k):
            if self.display_count >= k:
                self.display_idx = k
        
        if key is not None and key >= glfw.KEY_0 and key <= glfw.KEY_9:
            num = int(int(glfw_key(key)) + 10 * np.floor(int(glfw_key(key)) == 0)) # adds 10 only when key is 0
            set_display(num)
        
        # control modifier flags. We dont register these but these are used for activating mods
        if key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
            self.mods = glfw.MOD_SHIFT
        if key == glfw.KEY_LEFT_CONTROL or key == glfw.KEY_RIGHT_CONTROL:
            self.mods = glfw.MOD_CONTROL
        if key == glfw.KEY_LEFT_ALT or key == glfw.KEY_RIGHT_ALT:
            self.mods = glfw.MOD_ALT
        if key == glfw.KEY_CAPS_LOCK:
            self.mods = glfw.MOD_CAPS_LOCK
        if key == glfw.KEY_NUM_LOCK:
            self.mods = glfw.MOD_NUM_LOCK


    def key_release(self, key):
        pass

    #----------------------------------

    # handle mouse click inputs
    def mouse_click_callback(self, window, button, action, mods):
        
        if action == glfw.PRESS:
            self.mouse_is_clicked = True
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_mouse_down = True
                self.left_mouse_click()
            if button == glfw.MOUSE_BUTTON_RIGHT:
                self.right_mouse_down = True
                self.right_mouse_click()    
            if button == glfw.MOUSE_BUTTON_MIDDLE:
                self.middle_mouse_down = True
                self.middle_mouse_click()    
        elif action == glfw.RELEASE:
            self.mouse_is_clicked = False
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_mouse_down = False
                self.left_mouse_release()
            if button == glfw.MOUSE_BUTTON_RIGHT:
                self.right_mouse_down = False
                self.right_mouse_release()    
            if button == glfw.MOUSE_BUTTON_MIDDLE:
                self.middle_mouse_down = False
                self.middle_mouse_release()    
            
        self.click_position = np.array(glfw.get_cursor_pos(window))

    #----------------------------------

    # overwrite these functions with custom mouse events
    def left_mouse_click(self):
        pass

    def right_mouse_click(self):
        pass

    def middle_mouse_click(self):
        pass

    def left_mouse_release(self):
        pass

    def right_mouse_release(self):
        pass

    def middle_mouse_release(self):
        pass

    #----------------------------------

    # handle mouse move inputs
    def mouse_move_callback(self, window, xpos, ypos):
        self.mouse_is_moved = True
        move_position = np.array(glfw.get_cursor_pos(window))
        self.mouse_move(move_position)
        self.need_update = True
        

    #----------------------------------

    # overwrite this function with custom mouse move event
    def mouse_move(self, move_position):
        pass

    #----------------------------------

    # handle mouse scroll input
    def mouse_scroll_callback(self, window, x_offset, y_offset):
        self.mouse_scroll(y_offset)
        self.need_update = True

    #----------------------------------

    # overwrite this function with custom mouse scroll event
    def mouse_scroll(self, sign):
        pass


class Trackball3DWindow(OpenGLWindow):
    """
    An interactive OpenGL window with a 3D trackball controlled by mouse input. Drag with the left mouse button to
    rotate, drag with the right mouse button to move sideways, and scroll or drag with the middle mouse button to move
    towards or away from the anchor.

    Use this class as a base class for your own windows with custom functionality. Refer to

    Attributes
    ----------
    trackball : Trackball
        The current trackball state. Use `self.trackball.camera` to access the current camera state.
    initial_trackball : Trackball
        The trackball state used in the beginning or after having pressed X. Supply initial camera intrinsics here.
    lock_roll : bool
        If true, the camera's roll is fixed to 0, i.e., it's always upright.
    rotation_speed : float
        How fast the camera rotates when dragging with the left mouse button.
    sideways_speed : float
        How fast the camera translates left, right, up, and down when dragging with the right mouse button.
    forwards_speed : float
        How fast the camera translates towards and away from the current anchor when scrolling or dragging with the
        middle mouse button.
    """

    def __init__(
            self, display_res=(1, 1), title="Trackball3D Window", logging_level=logging.INFO, max_fps=None,
            initial_trackball=None, lock_roll=True, rotation_speed=0.005, sideways_speed=0.001, forwards_speed=0.2
    ):
        super().__init__(display_res, title, logging_level, max_fps)
        self._lock_roll = lock_roll
        self.rotation_speed = rotation_speed
        self.sideways_speed = sideways_speed
        self.forwards_speed = forwards_speed
        self.initial_trackball = Trackball() if initial_trackball is None else initial_trackball
        self.reset_trackball()

    def reset_trackball(self):
        """Resets the trackball state to that stored in the `initial_trackball` attribute."""
        self.trackball = deepcopy(self.initial_trackball)
        self._start_dragging()

    @property
    def lock_roll(self):
        return self._lock_roll

    @lock_roll.setter
    def lock_roll(self, lock_roll):
        self._lock_roll = lock_roll
        if lock_roll:
            self.trackball.camera.zero_roll()
        self._start_dragging()

    def _start_dragging(self):
        self.base_trackball = self.trackball
        self.click_position = np.array(glfw.get_cursor_pos(self.window_handle))

    left_mouse_click = _start_dragging
    right_mouse_click = _start_dragging
    middle_mouse_click = _start_dragging

    def mouse_move(self, move_position):
        if self.left_mouse_down or self.right_mouse_down or self.middle_mouse_down:
            mov = move_position - self.click_position
            self.trackball = deepcopy(self.base_trackball)
            if self.left_mouse_down:
                self.trackball.rotate(mov[0], -mov[1], self.rotation_speed)
                if self.lock_roll:
                    self.trackball.camera.zero_roll()
            elif self.right_mouse_down:
                self.trackball.translate_on_screen_plane(mov[0], -mov[1], self.sideways_speed)
            else:
                self.trackball.translate_into_screen_plane(-0.01 * mov[1], self.forwards_speed)

    def mouse_scroll(self, sign):
        self.trackball.translate_into_screen_plane(sign, self.forwards_speed)
        self._start_dragging()

    def key_press(self, key):
        super().key_press(key)
        if self.register_key(glfw.KEY_X, key, "Reset trackball."):
            self.reset_trackball()
        elif self.register_key(glfw.KEY_L, key, "Lock roll."):
            self.lock_roll = not self.lock_roll


def load_shader(file_path, defines=None):
    with open(file_path, "r") as file:
        source = file.read()
    define_pattern = re.compile(r"^\s*#define\s+(\w+)\s+(.*)$", re.MULTILINE)
    lines =  [line for line in source.splitlines() if line.strip()]
    existing_defines = {match.group(1): match.group(2) for match in define_pattern.finditer(source)}
    if defines:
        for key, value in defines.items():
            existing_defines[key] = value
    lines = [line for line in lines if not define_pattern.match(line)]
    define_lines = [f"#define {key} {value}" for key, value in existing_defines.items()]    
    if lines[0].startswith("#version"):
        processed_source = "\n".join([lines[0]] + define_lines + lines[1:])
    else:
        processed_source = "\n".join(define_lines + lines)
    return processed_source




# create a shader program, consisting of a vertex, fragment, and (optionally) geometry shader
def create_shader(path_vertex, path_geometry, path_fragment, defines=dict()):
    vertex_src = load_shader(path_vertex + ".vert", defines.get("vertex"))
    fragment_src = load_shader(path_fragment + ".frag", defines.get("fragment"))
    arg_list = [
        compileShader(vertex_src, GL_VERTEX_SHADER), 
        compileShader(fragment_src, GL_FRAGMENT_SHADER)]
    if path_geometry:
        geometry_src = load_shader(path_geometry + ".geom", defines.get("geometry"))
        arg_list.append(compileShader(geometry_src, GL_GEOMETRY_SHADER))
    return compileProgram(*arg_list)

class Rendertarget:
    def __init__(self):
        self.color = None
        self.depth = None


_cached_plugin = {}
def _get_plugin(gl=False):
    assert isinstance(gl, bool)

    # Return cached plugin if already loaded.
    if _cached_plugin.get(gl, None) is not None:
        return _cached_plugin[gl]

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        lib_dir = os.path.dirname(__file__) + r"\..\lib"
        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                vs_relative_path = r"\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition
                paths = sorted(glob.glob(r"C:\Program Files" + vs_relative_path), reverse=True)
                paths += sorted(glob.glob(r"C:\Program Files (x86)" + vs_relative_path), reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

    # Compiler options.
    opts = ['-DTORCH', '-g']

    # Linker options for the GL-interfacing plugin.
    ldflags = []
    if gl:
        if os.name == 'posix':
            ldflags = ['-lGL', '-lEGL']
        elif os.name == 'nt':
            libs = ['gdi32', 'opengl32', 'user32', 'setgpu']
            ldflags = ['/LIBPATH:' + lib_dir] + ['/DEFAULTLIB:' + x for x in libs]

    # List of source files.
    source_files = [
        'common/common.cpp',
        'common/texture.cu',
        'torch_bindings.cpp',
        'torch_gl_ops.cpp'
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # On Linux, show a warning if GLEW is being forcibly loaded when compiling the GL plugin.
    if gl and (os.name == 'posix') and ('libGLEW' in os.environ.get('LD_PRELOAD', '')):
        logging.getLogger('ismael').warning("Warning: libGLEW is being loaded via LD_PRELOAD, and will probably conflict with the OpenGL plugin")

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    plugin_name = 'ismael_plugin' + ('_gl' if gl else '')
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory(plugin_name, False), 'lock')
        if os.path.exists(lock_fn):
            logging.getLogger('ismael').warning("Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Speed up compilation on Windows.
    if os.name == 'nt':
        # Skip telemetry sending step in vcvarsall.bat
        os.environ['VSCMD_SKIP_SENDTELEMETRY'] = '1'

        # Opportunistically patch distutils to cache MSVC environments.
        try:
            import distutils._msvccompiler
            import functools
            if not hasattr(distutils._msvccompiler._get_vc_env, '__wrapped__'):
                distutils._msvccompiler._get_vc_env = functools.lru_cache()(distutils._msvccompiler._get_vc_env)
        except:
            pass

    # Compile and load. # Set verbose=True for debugging messages.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name=plugin_name, sources=source_paths, extra_cflags=opts, extra_cuda_cflags=opts+['-lineinfo'], extra_ldflags=ldflags, with_cuda=True, verbose=False)

    # Import, cache, and return the compiled module.
    _cached_plugin[gl] = importlib.import_module(plugin_name)
    print("ismael plugin successfully built...")
    return _cached_plugin[gl]

class CopyTextureToTensor():
    def __init__(self, tex):
        self.texture = tex
        assert self.texture.target in [GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY]

        self.handle = self.texture.handle
        self.width, self.height = self.texture.resolution
        self.channels = self.texture.channels
        self.ori_channels = None

        if self.texture.target == GL_TEXTURE_2D:
            self.layers = 1
        else:
            self.layers = self.texture._layers

        if self.channels == 3: # workaround for RGB image. cudaGraphicsGLRegisterImage doesnt support RGB
            self.ori_channels = self.channels
            self.channels = 4
        
        # Create empty buffer for storing the resulting tensor
        self.tensor = (torch.empty((self.layers, self.width, self.height, self.channels))).cuda()
        
        self.register_texture()
    
    def register_texture(self):
        # Register an OpenGL texture or renderbuffer object. We do this only once per texture.
        # WARNING: Destroying and creating this over and over may cause memory fragmentation
        self.graphics_resource, self.p = _get_plugin().RegisterTexture(
            self.handle,
            self.layers, self.width, self.height, self.channels,
            self.tensor)
        
    def copy_to_tensor(self):
        _get_plugin().TextureToTensor(self.graphics_resource, self.p)
        
        return_tensor = self.tensor.permute(0, 3, 1, 2).flip(2)
        
        if self.ori_channels == 3:
            return_tensor = return_tensor[:, 0:3, :, :]
        
        return return_tensor

def copy_tensor_to_texture(texture, tensor):
    # Input is a tensor of shape (batch=N layers, channels, width, height). If more than one tensor, we return a texture array (layered)
    assert isinstance(tensor, torch.Tensor)    
    assert tensor.is_cuda

    ori_channels = tensor.size(1)
    num_layers = tensor.size(0)

    # permute to make it (layers, h, w, c). # the same shape can/should be observed in the cpp function as well
    # and flip to account for the height flip in opengl. Cross-check: download_texture method in Texture2D
    tensor = tensor.permute(0, 2, 3, 1).flip(1)
    
    w, h, channels = tensor.shape[2], tensor.shape[1], tensor.shape[3] # (width, height, channels)
    texture_layout = (w, h)

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    handle = texture.handle
    # allocate memory such that the resulting texture is in the right layout (w, h, c)
    if texture.target == GL_TEXTURE_2D_ARRAY:
        assert num_layers >= 1        
        texture.allocate_memory(texture_layout, layers=num_layers, channels=channels)
    if texture.target == GL_TEXTURE_2D:
        assert num_layers == 1
        texture.allocate_memory(texture_layout, channels=channels)

    _get_plugin().TensorToTexture(tensor, handle)

    if ori_channels == 3:
        # remove the alpha channel from texture
        texture.gl_format = texture.gl_format_from_channel_count(ori_channels)
        texture.channels = ori_channels


         
