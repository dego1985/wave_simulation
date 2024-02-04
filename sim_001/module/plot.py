import numpy as np
from glumpy import app, gl, glm, gloo
import torch

import module.gpu_work as gw


class mesh():
    def __init__(self, motion):
        # plane
        self.motion = motion
        self.N = N = motion.N

        # vertices
        x = np.arange(N) / N * 2 - 1
        y = np.arange(N) / N * 2 - 1
        z = motion.update_numpy()
        x, y = np.meshgrid(x, y)

        vertices = np.transpose([x, y, z], (1, 2, 0)).reshape(-1, 3)

        # colors
        colors = np.random.randn(len(vertices), 4).astype(np.float32)

        # outline
        idx = []
        for i in np.arange(N-1):
            for j in np.arange(N-1):
                offset = i * N + j
                idx.append([offset, offset+1, offset+1+N, offset+N] +
                           [offset, offset+N, offset+1, offset+1+N])
        outline = np.array(idx).reshape(-1)

        # glumpy Vertex Buffer
        dtype = [("position", np.float32, 3),
                 ("color",    np.float32, 4)]
        VertexBuffer = np.zeros(len(vertices), dtype)
        VertexBuffer["position"] = vertices
        VertexBuffer["color"] = colors
        VertexBuffer = VertexBuffer.view(gloo.VertexBuffer)

        # glumpy Index Buffer
        outline = outline.astype(np.uint32).view(gloo.IndexBuffer)

        self.VertexBuffer = VertexBuffer
        self.outline = outline

        # torch
        v = torch.from_numpy(np.transpose(vertices, (1, 0)).reshape(1, 3, N, N).astype(np.float32)).cuda()
        c = torch.from_numpy(np.transpose(colors, (1, 0)).reshape(1, 4, N, N).astype(np.float32)).cuda()
        self.v = v
        self.c = c
    
    def update(self, dt=0):
        motion = self.motion
        v = self.v
        c = self.c

        z = motion.update(dt)
        c[0, 0] = 3*z
        c[0, 1] = -3*z
        c[0, 2] = z+1
        c[0, 3] = 1

        v[0, 2] = z


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
        program = gloo.Program(vertex, fragment)

        program.bind(VertexBuffer)
        program['model'] = np.eye(4, dtype=np.float32)
        program['view'] = glm.translation(0, 0, -2)

        VertexBuffer.activate()
        VertexBuffer.deactivate()

        self.RegisteredBuffer = gw.make_RegisteredBuffer(VertexBuffer)
        self.program = program
        self.outline = outline
    
    def update_VertexBuffer(self, dt=0):
        # torch
        self.obj.update(dt)
        v = self.obj.v
        c = self.obj.c
        V_ = torch.cat((v, c), dim=1)
        V_ = V_.contiguous(memory_format=torch.channels_last)

        # copy
        gw.copy_torch2RegisteredBuffer(self.RegisteredBuffer, V_[0])
        # gw.copy_torch2glumpy(VertexBuffer, V_[0])

    def on_draw(self, dt):
        program = self.program
        window = self.window

        # set title
        # window.set_title(str(window.fps).encode("ascii"))
        window.set_title(f"{window.fps:0.2f}")
        # tr(binascii.hexlify(header), 'utf-8')
        self.update_VertexBuffer(dt)

        # Filled cube
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
        self.phi += 10*dt  # degrees
        model = np.eye(4, dtype=np.float32)
        glm.rotate(model, 90, 1, 0, 0)
        glm.rotate(model, self.theta, 0, 0, 1)
        glm.rotate(model, self.phi, 0, 1, 0)
        glm.rotate(model, 45, 1, 0, 0)
        program['model'] = model

