package id.ac.comvis.yolo26depth

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImagePreprocessor {
    fun depthInput(bitmap: Bitmap, config: ModelConfig): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(
            bitmap,
            config.depthInputWidth,
            config.depthInputHeight,
            true,
        )
        val pixels = IntArray(config.depthInputWidth * config.depthInputHeight)
        resized.getPixels(
            pixels,
            0,
            config.depthInputWidth,
            0,
            0,
            config.depthInputWidth,
            config.depthInputHeight,
        )
        val buffer = ByteBuffer.allocateDirect(pixels.size * 3 * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        if (config.depthInputLayout == "NCHW") {
            for (channel in 0..2) {
                for (pixel in pixels) {
                    val value = when (channel) {
                        0 -> (pixel shr 16) and 0xff
                        1 -> (pixel shr 8) and 0xff
                        else -> pixel and 0xff
                    }
                    buffer.putFloat(value / 255f)
                }
            }
        } else {
            for (pixel in pixels) {
                buffer.putFloat(((pixel shr 16) and 0xff) / 255f)
                buffer.putFloat(((pixel shr 8) and 0xff) / 255f)
                buffer.putFloat((pixel and 0xff) / 255f)
            }
        }
        buffer.rewind()
        if (resized !== bitmap) resized.recycle()
        return buffer
    }

    fun detectorInput(
        bitmap: Bitmap,
        inputSize: Int,
    ): Pair<ByteBuffer, LetterboxTransform> {
        val scale = minOf(
            inputSize.toFloat() / bitmap.width,
            inputSize.toFloat() / bitmap.height,
        )
        val scaledWidth = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val scaledHeight = (bitmap.height * scale).toInt().coerceAtLeast(1)
        val padX = (inputSize - scaledWidth) / 2
        val padY = (inputSize - scaledHeight) / 2
        val scaled = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
        val letterboxed = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        letterboxed.eraseColor(0xff727272.toInt())
        val canvas = android.graphics.Canvas(letterboxed)
        canvas.drawBitmap(scaled, padX.toFloat(), padY.toFloat(), null)
        val pixels = IntArray(inputSize * inputSize)
        letterboxed.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val planeSize = inputSize * inputSize
        val buffer = ByteBuffer.allocateDirect(planeSize * 3 * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        for (channel in 0..2) {
            for (pixel in pixels) {
                val value = when (channel) {
                    0 -> (pixel shr 16) and 0xff
                    1 -> (pixel shr 8) and 0xff
                    else -> pixel and 0xff
                }
                buffer.putFloat(value / 255f)
            }
        }
        buffer.rewind()
        if (scaled !== bitmap) scaled.recycle()
        letterboxed.recycle()
        return buffer to LetterboxTransform(
            scale = scale,
            padX = padX.toFloat(),
            padY = padY.toFloat(),
            originalWidth = bitmap.width,
            originalHeight = bitmap.height,
        )
    }
}
