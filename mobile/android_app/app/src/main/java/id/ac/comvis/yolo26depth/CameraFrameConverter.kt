package id.ac.comvis.yolo26depth

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

object CameraFrameConverter {
    fun toUprightBitmap(image: ImageProxy): Bitmap {
        require(image.format == android.graphics.ImageFormat.YUV_420_888) {
            "CameraX frame must use YUV_420_888; got format=${image.format}."
        }
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuffer = yPlane.buffer.duplicate()
        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()

        val width = image.width
        val height = image.height
        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val yIndex = y * yPlane.rowStride + x * yPlane.pixelStride
                val uvX = x / 2
                val uvY = y / 2
                val uIndex = uvY * uPlane.rowStride + uvX * uPlane.pixelStride
                val vIndex = uvY * vPlane.rowStride + uvX * vPlane.pixelStride
                val yy = (yBuffer.get(yIndex).toInt() and 0xff) - 16
                val uu = (uBuffer.get(uIndex).toInt() and 0xff) - 128
                val vv = (vBuffer.get(vIndex).toInt() and 0xff) - 128
                val luminance = (yy.coerceAtLeast(0) * 1192)
                val red = ((luminance + 1634 * vv) shr 10).coerceIn(0, 255)
                val green = ((luminance - 833 * vv - 400 * uu) shr 10).coerceIn(0, 255)
                val blue = ((luminance + 2066 * uu) shr 10).coerceIn(0, 255)
                pixels[y * width + x] =
                    (0xff shl 24) or (red shl 16) or (green shl 8) or blue
            }
        }
        val full = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        // Keep the CameraX crop in sensor coordinates, then rotate once into display coordinates.
        val crop = image.cropRect
        val cropped = if (
            crop.left == 0 && crop.top == 0 && crop.right == width && crop.bottom == height
        ) {
            full
        } else {
            Bitmap.createBitmap(
                full,
                crop.left,
                crop.top,
                crop.width(),
                crop.height(),
            ).also { full.recycle() }
        }
        val rotation = image.imageInfo.rotationDegrees
        require(rotation % 90 == 0) {
            "CameraX rotation must be a multiple of 90 degrees; got $rotation."
        }
        if (rotation == 0) return cropped
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(
            cropped,
            0,
            0,
            cropped.width,
            cropped.height,
            matrix,
            true,
        ).also { cropped.recycle() }
    }
}
