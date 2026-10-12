package id.ac.comvis.yolo26depth

import kotlin.math.roundToInt

class DepthEma(private val alpha: Float) {
    private var previous: FloatArray? = null
    private var previousWidth = 0
    private var previousHeight = 0

    fun update(
        current: FloatArray,
        width: Int = current.size,
        height: Int = 1,
    ): FloatArray {
        require(width > 0 && height > 0 && current.size == width * height)
        val old = previous
        val result = if (
            old == null || old.size != current.size ||
            previousWidth != width || previousHeight != height
        ) {
            current.copyOf()
        } else {
            FloatArray(current.size) { index ->
                alpha * current[index] + (1f - alpha) * old[index]
            }
        }
        previous = result
        previousWidth = width
        previousHeight = height
        return result.copyOf()
    }

    fun reset() {
        previous = null
        previousWidth = 0
        previousHeight = 0
    }
}

object DepthMath {
    fun removeLetterbox(
        depthMeters: FloatArray,
        modelWidth: Int,
        modelHeight: Int,
        transform: LetterboxTransform,
    ): CroppedDepthMap {
        require(depthMeters.size == modelWidth * modelHeight)
        require(modelWidth > 0 && modelHeight > 0)
        val width = (transform.originalWidth * transform.scale).roundToInt()
            .coerceIn(1, modelWidth)
        val height = (transform.originalHeight * transform.scale).roundToInt()
            .coerceIn(1, modelHeight)
        val left = transform.padX.toInt().coerceIn(0, modelWidth - width)
        val top = transform.padY.toInt().coerceIn(0, modelHeight - height)
        val cropped = FloatArray(width * height)
        for (row in 0 until height) {
            depthMeters.copyInto(
                destination = cropped,
                destinationOffset = row * width,
                startIndex = (top + row) * modelWidth + left,
                endIndex = (top + row) * modelWidth + left + width,
            )
        }
        return CroppedDepthMap(cropped, width, height)
    }

    fun sampleMedianInBox(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
    ): Float? {
        require(depthMeters.size == width * height)
        if (width <= 0 || height <= 0 || right <= left || bottom <= top) return null

        val insetX = (right - left) * 0.2f
        val insetY = (bottom - top) * 0.2f
        val startX = (left + insetX).toInt().coerceIn(0, width - 1)
        val endX = kotlin.math.ceil(right - insetX).toInt().coerceIn(startX + 1, width)
        val startY = (top + insetY).toInt().coerceIn(0, height - 1)
        val endY = kotlin.math.ceil(bottom - insetY).toInt().coerceIn(startY + 1, height)
        val values = ArrayList<Float>((endX - startX) * (endY - startY))

        for (py in startY until endY) {
            for (px in startX until endX) {
                val value = depthMeters[py * width + px]
                if (value.isFinite() && value > 0f) values += value
            }
        }
        if (values.isEmpty()) return null
        values.sort()
        return values[values.size / 2]
    }

    fun sampleMedian(
        depthMeters: FloatArray,
        width: Int,
        height: Int,
        x: Float,
        y: Float,
        patchSize: Int,
    ): Float? {
        val centerX = x.roundToInt()
        val centerY = y.roundToInt()
        if (centerX !in 0 until width || centerY !in 0 until height) return null
        val radius = patchSize / 2
        val values = ArrayList<Float>(patchSize * patchSize)
        for (py in (centerY - radius).coerceAtLeast(0)..(centerY + radius).coerceAtMost(height - 1)) {
            for (px in (centerX - radius).coerceAtLeast(0)..(centerX + radius).coerceAtMost(width - 1)) {
                val value = depthMeters[py * width + px]
                if (value.isFinite() && value > 0f) values += value
            }
        }
        if (values.isEmpty()) return null
        values.sort()
        return values[values.size / 2]
    }
}
