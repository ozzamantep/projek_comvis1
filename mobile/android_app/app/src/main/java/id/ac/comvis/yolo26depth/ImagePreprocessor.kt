package id.ac.comvis.yolo26depth

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

object ImagePreprocessor {
    fun multitaskInput(
        bitmap: Bitmap,
        inputWidth: Int,
        inputHeight: Int,
    ): Pair<ByteBuffer, LetterboxTransform> {
        val scale = minOf(
            inputWidth.toFloat() / bitmap.width,
            inputHeight.toFloat() / bitmap.height,
        )
        val scaledWidth = (bitmap.width * scale).roundToInt().coerceAtLeast(1)
        val scaledHeight = (bitmap.height * scale).roundToInt().coerceAtLeast(1)
        val padX = (inputWidth - scaledWidth) / 2
        val padY = (inputHeight - scaledHeight) / 2
        val scaled = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
        val letterboxed = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        letterboxed.eraseColor(0xff727272.toInt())
        val canvas = android.graphics.Canvas(letterboxed)
        canvas.drawBitmap(scaled, padX.toFloat(), padY.toFloat(), null)
        val pixels = IntArray(inputWidth * inputHeight)
        letterboxed.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        val planeSize = inputWidth * inputHeight
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
