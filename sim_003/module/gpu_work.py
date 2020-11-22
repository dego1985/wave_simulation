from contextlib import contextmanager
import time

import numpy as np
import pycuda.autoinit
import pycuda
import pycuda.gl
import pycuda.gpuarray as gp
from pycuda.gl import graphics_map_flags
import torch
from glumpy import app, gl, glm, gloo


@contextmanager
def cuda_activate(cuda_buffer):
    """Context manager simplifying use of pycuda.gl.RegisteredBuffer"""
    mapping = cuda_buffer.map()
    ptr, size = mapping.device_ptr_and_size()
    yield ptr
    mapping.unmap()


def glumpy_pointer(
    buf: gloo.VertexBuffer,
):
    cuda_buffer = pycuda.gl.RegisteredBuffer(
        int(buf.handle), graphics_map_flags.WRITE_DISCARD)
    mapping = cuda_buffer.map()
    ptr, size = mapping.device_ptr_and_size()
    mapping.unmap()
    return ptr


def clone_glumpy2pycuda(
    src: gloo.VertexBuffer,
):
    # glumpy 2 pycuda
    cuda_buffer = pycuda.gl.RegisteredBuffer(
        int(src.handle), graphics_map_flags.WRITE_DISCARD)
    with cuda_activate(cuda_buffer) as ptr:
        dst_pycuda = gp.GPUArray(src.shape, np.float32, gpudata=ptr).copy()

    return dst_pycuda


def make_RegisteredBuffer(
    src: gloo.VertexBuffer,
) -> pycuda.gl.RegisteredBuffer:
    # pycuda 2 glumpy
    RegisteredBuffer = pycuda.gl.RegisteredBuffer(
        int(src.handle), graphics_map_flags.WRITE_DISCARD)
    return RegisteredBuffer


def copy_pycuda2glumpy(
    dst: gloo.VertexBuffer,
    src: gp.GPUArray
):
    # pycuda 2 glumpy
    RegisteredBuffer = pycuda.gl.RegisteredBuffer(
        int(dst.handle), graphics_map_flags.WRITE_DISCARD)
    copy_pycuda2RegisteredBuffer(RegisteredBuffer, src)
    return


def copy_pycuda2RegisteredBuffer(
    dst: pycuda.gl.RegisteredBuffer,
    src: gp.GPUArray
):
    # pycuda 2 RegisteredBuffer
    with cuda_activate(dst) as ptr:
        dst = gp.GPUArray(src.shape, np.float32, gpudata=ptr)
        dst[:] = src
    return


def copy_torch2glumpy(
    dst: gloo.VertexBuffer,
    src: torch.cuda.FloatTensor
):
    # torch 2 pycuda
    src = gp.GPUArray(src.shape, np.float32, gpudata=src.data_ptr())

    # copy pycuda 2 glumpy
    copy_pycuda2glumpy(dst, src)
    return


def copy_torch2RegisteredBuffer(
    dst: pycuda.gl.RegisteredBuffer,
    src: torch.cuda.FloatTensor
):
    # torch 2 pycuda
    src = gp.GPUArray(src.shape, np.float32, gpudata=src.data_ptr())

    # copy pycuda 2 glumpy
    copy_pycuda2RegisteredBuffer(dst, src)
    return

if __name__ == "__main__":
    assert torch.cuda.is_available()

    # torch
    shape = (10000, 10000)
    x = torch.ones([shape[0], shape[1]], dtype=torch.float32).cuda()

    # glumpy mem aloc
    window = app.Window()
    y = np.zeros(shape, np.float32)
    y = y.view(gloo.VertexBuffer)
    y.activate()
    y.deactivate()

    copy_torch2glumpy(y, x)
    z = clone_glumpy2pycuda(y)

    print(z)

