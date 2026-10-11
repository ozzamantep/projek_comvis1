package id.ac.comvis.yolo26depth

object InfernoPalette {
    private val anchors = intArrayOf(
        0x000004, 0x1b0c41, 0x4a0c6b, 0x781c6d, 0xa52c60,
        0xcf4446, 0xed6925, 0xfb9b06, 0xf7d13d, 0xfcffa4,
    )

    fun colorizeForDisplay(depthMeters: FloatArray): IntArray {
        val valid = depthMeters.filter { it.isFinite() && it > 0f }.sorted()
        if (valid.isEmpty()) return IntArray(depthMeters.size) { 0xff000000.toInt() }
        val low = percentile(valid, 0.02f)
        val high = percentile(valid, 0.98f)
        if (high - low < 1e-6f) {
            return IntArray(depthMeters.size) { 0xff000000.toInt() }
        }
        return IntArray(depthMeters.size) { index ->
            val value = depthMeters[index]
            if (!value.isFinite() || value <= 0f) {
                0xff000000.toInt()
            } else {
                val normal = (1f - ((value - low) / (high - low)).coerceIn(0f, 1f))
                interpolate(normal)
            }
        }
    }

    private fun percentile(sorted: List<Float>, fraction: Float): Float {
        val position = fraction * (sorted.size - 1)
        val lower = position.toInt()
        val upper = (lower + 1).coerceAtMost(sorted.lastIndex)
        val weight = position - lower
        return sorted[lower] * (1f - weight) + sorted[upper] * weight
    }

    private fun interpolate(value: Float): Int {
        val position = value * (anchors.size - 1)
        val index = position.toInt().coerceIn(0, anchors.lastIndex - 1)
        val fraction = position - index
        val a = anchors[index]
        val b = anchors[index + 1]
        fun channel(shift: Int): Int {
            val av = (a shr shift) and 0xff
            val bv = (b shr shift) and 0xff
            return (av + (bv - av) * fraction).toInt().coerceIn(0, 255)
        }
        return (0xff shl 24) or
            (channel(16) shl 16) or
            (channel(8) shl 8) or
            channel(0)
    }
}
