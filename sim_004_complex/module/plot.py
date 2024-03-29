import numpy as np
from glumpy import app, gl, glm, gloo
import torch

import module.gpu_work as gw


class mesh():
    def __init__(self, motion):
        # plane
        self.motion = motion
        self.N = N = motion.N[:2]
        self.dx = dx = motion.dx

        # vertices
        X = [dx * (np.arange(N[i]) - N[i] * 0.5) for i in range(2)]
        x, y = X
        x, y = np.meshgrid(x, y)
        z = motion.update_numpy()

        vertices = np.transpose([x, y, z], (1, 2, 0)).reshape(-1, 3)

        # colors
        colors = np.random.randn(len(vertices), 4).astype(np.float32)

        # outline
        idx = []
        for i in np.arange(N[1]-1):
            for j in np.arange(N[0]-1):
                offset = i * N[0] + j
                idx.append([offset, offset+1, offset+1+N[0], offset+N[0]] +
                           [offset, offset+N[0], offset+1, offset+1+N[0]])
        outline = np.array(idx).reshape(-1)

        # outline
        idx = np.arange(N[0]*N[1])
        point_idx = np.array(idx).reshape(-1)

        ############################################################
        # glumpy Vertex Buffer
        dtype = [("position", np.float32, 3),
                 ("color",    np.float32, 4)]
        VertexBuffer = np.zeros(len(vertices), dtype)
        VertexBuffer["position"] = vertices
        VertexBuffer["color"] = colors
        VertexBuffer = VertexBuffer.view(gloo.VertexBuffer)

        # glumpy Index Buffer
        outline = outline.astype(np.uint32).view(gloo.IndexBuffer)

        # glumpy Index Buffer
        point_idx = point_idx.astype(np.uint32).view(gloo.IndexBuffer)

        ############################################################
        # self
        self.VertexBuffer = VertexBuffer
        self.outline = outline
        self.point_idx = point_idx

        ############################################################
        # torch
        v = torch.from_numpy(np.transpose(vertices, (1, 0)).reshape(1, 3, N[0], N[1]).astype(np.float32)).cuda()
        c = torch.from_numpy(np.transpose(colors, (1, 0)).reshape(1, 4, N[0], N[1]).astype(np.float32)).cuda()
        self.v = v
        self.c = c

    def update(self, dt=0):
        motion = self.motion
        v = self.v
        c = self.c

        z = motion.update(dt)

        zc = 0.5 * z
        c[0, 0] = 0 + 2*zc
        c[0, 1] = 0.5 - zc
        c[0, 2] = 1.0 + 2*zc
        c[0, 3] = 1

        v[0, 2] = z*0.3


class plot3d():
    def __init__(self, obj):
        self.obj = obj
        self.phi, self.theta = 0, 0

        # init
        self.init_window()
        self.bind_obj(obj)
        self.update_VertexBuffer()
        app.run()

    def init_window(self):
        window = app.Window(width=1920, height=1080,
                            color=(0.30, 0.30, 0.35, 1.00))

        @window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glPolygonOffset(1, 1)
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glLineWidth(0.55)

        @window.event
        def on_draw(dt):
            window.clear()
            self.on_draw(dt)

        @window.event
        def on_resize(width, height):
            program = self.program
            program['projection'] = glm.perspective(
                45.0, width / float(height), 0.1, 100.0)

        self.window = window

    def bind_obj(self, obj):
        # make obj
        vertex = """
        uniform vec4 ucolor;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        attribute vec3 position;
        attribute vec4 color;
        varying vec4 v_color;
        void main()
        {
            v_color = ucolor * color;
            gl_Position = projection * view * model * vec4(position,1.0);
        }
        """

        fragment = """
        varying vec4 v_color;
        void main()
        {
            gl_FragColor = v_color;
        }
        """

        VertexBuffer = obj.VertexBuffer
        outline = obj.outline
        point_idx = obj.point_idx
        program = gloo.Program(vertex, fragment)

        program.bind(VertexBuffer)
        program['model'] = np.eye(4, dtype=np.float32)
        program['view'] = glm.translation(0, 0, -5)

        VertexBuffer.activate()
        VertexBuffer.deactivate()

        self.RegisteredBuffer = gw.make_RegisteredBuffer(VertexBuffer)
        self.program = program
        self.outline = outline
        self.point_idx = point_idx
    
    def update_VertexBuffer(self, dt=0):
        # torch
        self.obj.update(dt)
        v = self.obj.v
        c = self.obj.c
        V_ = torch.cat((v, c), dim=1)
        V_ = V_.contiguous(memory_format=torch.channels_last)

        # copy
        gw.copy_torch2RegisteredBuffer(self.RegisteredBuffer, V_[0])

    def on_draw(self, dt):
        program = self.program
        window = self.window

        # set title
        window.set_title(f"{window.fps:0.2f}")


        self.update_VertexBuffer(dt)

        # # Point
        # gl.glDisable(gl.GL_BLEND)
        # gl.glEnable(gl.GL_DEPTH_TEST)
        # gl.glPointSize(5)
        # program['ucolor'] = 1, 1, 1, 1
        # program.draw(gl.GL_POINTS, self.point_idx)

        # Fill
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        program['ucolor'] = 1, 1, 1, 1
        program.draw(gl.GL_QUADS, self.outline)

        # Outlined program
        # gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        # gl.glEnable(gl.GL_BLEND)
        # gl.glDepthMask(gl.GL_FALSE)
        # program['ucolor'] = 0, 0, 0, 1
        # program.draw(gl.GL_LINES, self.outline)
        # gl.glDepthMask(gl.GL_TRUE)

        # Make program rotate
        self.theta += 0*dt  # degrees
        self.phi += 2*dt  # degrees
        model = np.eye(4, dtype=np.float32)
        glm.rotate(model, -90, 1, 0, 0)
        glm.rotate(model, self.theta, 0, 0, 1)
        glm.rotate(model, self.phi, 0, 1, 0)
        glm.rotate(model, 45, 1, 0, 0)
        program['model'] = model

