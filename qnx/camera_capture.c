#include "camera_capture.h"

#include <camera/camera_api.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    Frame frame;
    size_t capacity;
} FrameBuffer;

struct MedusaCamera {
    camera_handle_t handle;
    pthread_mutex_t lock;
    FrameBuffer pending;
    FrameBuffer delivered;
    bool pending_ready;
};

static void receive_frame(camera_handle_t handle, camera_buffer_t *buffer, void *context)
{
    MedusaCamera *camera = context;
    uint32_t width, height, stride, format;
    size_t bytes_per_pixel;
    (void)handle;

    switch (buffer->frametype) {
    case CAMERA_FRAMETYPE_YCBYCR:
        width = buffer->framedesc.ycbycr.width;
        height = buffer->framedesc.ycbycr.height;
        stride = buffer->framedesc.ycbycr.stride;
        format = PIXEL_YUYV;
        bytes_per_pixel = 2;
        break;
    case CAMERA_FRAMETYPE_CBYCRY:
        width = buffer->framedesc.cbycry.width;
        height = buffer->framedesc.cbycry.height;
        stride = buffer->framedesc.cbycry.stride;
        format = PIXEL_UYVY;
        bytes_per_pixel = 2;
        break;
    case CAMERA_FRAMETYPE_RGB8888:
        width = buffer->framedesc.rgb8888.width;
        height = buffer->framedesc.rgb8888.height;
        stride = buffer->framedesc.rgb8888.stride;
        format = PIXEL_RGBX;
        bytes_per_pixel = 4;
        break;
    case CAMERA_FRAMETYPE_BGR8888:
        width = buffer->framedesc.rgb8888.width;
        height = buffer->framedesc.rgb8888.height;
        stride = buffer->framedesc.rgb8888.stride;
        format = PIXEL_BGRX;
        bytes_per_pixel = 4;
        break;
    default:
        return;
    }

    /* framesize describes the camera_buffer_t struct. The descriptor gives the pixel layout. */
    if (!buffer->framebuf || !stride || !width || !height
        || width > SIZE_MAX / bytes_per_pixel || height > SIZE_MAX / stride
        || (bytes_per_pixel == 2 && width % 2 != 0)) {
        return;
    }
    size_t row_bytes = (size_t)width * bytes_per_pixel;
    if (row_bytes > stride) {
        return;
    }

    size_t frame_bytes = row_bytes * height;
    pthread_mutex_lock(&camera->lock);
    FrameBuffer *pending = &camera->pending;
    if (frame_bytes > pending->capacity) {
        uint8_t *data = realloc(pending->frame.data, frame_bytes);
        if (!data) {
            pthread_mutex_unlock(&camera->lock);
            return;
        }
        pending->frame.data = data;
        pending->capacity = frame_bytes;
    }
    for (uint32_t row = 0; row < height; ++row) {
        memcpy(pending->frame.data + row * row_bytes,
               buffer->framebuf + (size_t)row * stride, row_bytes);
    }
    pending->frame.width = width;
    pending->frame.height = height;
    pending->frame.format = format;
    camera->pending_ready = true;
    pthread_mutex_unlock(&camera->lock);
}

static void free_camera(MedusaCamera *camera)
{
    pthread_mutex_destroy(&camera->lock);
    free(camera->pending.frame.data);
    free(camera->delivered.frame.data);
    free(camera);
}

static int configure_viewfinder(camera_handle_t handle)
{
    camera_frametype_t available[CAMERA_FRAMETYPE_NUMFRAMETYPES];
    const camera_frametype_t preferred[] = {
        CAMERA_FRAMETYPE_YCBYCR, CAMERA_FRAMETYPE_CBYCRY,
        CAMERA_FRAMETYPE_RGB8888, CAMERA_FRAMETYPE_BGR8888,
    };
    uint32_t count = 0;
    int error = camera_get_supported_vf_frame_types(
        handle, CAMERA_FRAMETYPE_NUMFRAMETYPES, &count, available);
    if (error != EOK) {
        return error;
    }
    for (size_t i = 0; i < sizeof(preferred) / sizeof(preferred[0]); ++i) {
        for (uint32_t j = 0; j < count; ++j) {
            if (available[j] == preferred[i]) {
                return camera_set_vf_property(handle, CAMERA_IMGPROP_FORMAT, preferred[i],
                                              CAMERA_IMGPROP_CREATEWINDOW, 0);
            }
        }
    }
    return ENOTSUP;
}

MedusaCamera *medusa_camera_open(int unit, int *error)
{
    MedusaCamera *camera = calloc(1, sizeof(*camera));
    if (!camera) {
        *error = ENOMEM;
        return NULL;
    }
    *error = pthread_mutex_init(&camera->lock, NULL);
    if (*error != 0) {
        free(camera);
        return NULL;
    }

    *error = camera_open((camera_unit_t)unit, CAMERA_MODE_RO, &camera->handle);
    if (*error != EOK) {
        free_camera(camera);
        return NULL;
    }
    *error = configure_viewfinder(camera->handle);
    if (*error == EOK) {
        *error = camera_start_viewfinder(camera->handle, receive_frame, NULL, camera);
    }
    if (*error != EOK) {
        camera_close(camera->handle);
        free_camera(camera);
        return NULL;
    }
    return camera;
}

/* Only the reader touches the delivered buffer until it reads again. */
int medusa_camera_read(MedusaCamera *camera, Frame *frame)
{
    pthread_mutex_lock(&camera->lock);
    bool ready = camera->pending_ready;
    if (ready) {
        FrameBuffer previous = camera->delivered;
        camera->delivered = camera->pending;
        camera->pending = previous;
        camera->pending_ready = false;
        *frame = camera->delivered.frame;
    }
    pthread_mutex_unlock(&camera->lock);
    return ready;
}

void medusa_camera_close(MedusaCamera *camera)
{
    camera_stop_viewfinder(camera->handle);
    camera_close(camera->handle);
    free_camera(camera);
}
