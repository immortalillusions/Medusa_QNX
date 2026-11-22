#ifndef MEDUSA_CAMERA_CAPTURE_H
#define MEDUSA_CAMERA_CAPTURE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum PixelFormat {
    PIXEL_YUYV = 1,
    PIXEL_UYVY,
    PIXEL_RGBX,
    PIXEL_BGRX,
};

typedef struct MedusaCamera MedusaCamera;

/* Shared layout with the Python ctypes client. */
typedef struct {
    uint8_t *data;
    uint32_t width;
    uint32_t height;
    uint32_t format;
} Frame;

MedusaCamera *medusa_camera_open(int unit, int *error);
/* The returned data is valid until the next successful read or close. */
int medusa_camera_read(MedusaCamera *camera, Frame *frame);
void medusa_camera_close(MedusaCamera *camera);

#ifdef __cplusplus
}
#endif

#endif
