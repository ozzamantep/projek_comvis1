package id.ac.comvis.yolo26depth

import kotlin.math.roundToInt

class DepthEma(private val alpha: Float) {
    private var previous: FloatArray? = null

    fun update(current: FloatArray): FloatArray {
        val old = previous
        val result = if (old == null || old.size != current.size) {
            current.copyOf()
        } else {
            FloatArray(current.size) { index ->
                alpha * current[index] + (1f - alpha) * old[index]
            }
        }
        previous = result
        return result.copyOf()
    }

    fun reset() {
        previous = null
    }
}

object DepthMath {
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
