package id.ac.comvis.yolo26depth

import kotlin.math.floor
import kotlin.math.sqrt

data class MarkerCalibrationFit(
    val calibration: LinearDepthCalibration,
    val pairs: List<CalibrationPair>,
    val rSquared: Float,
    val rmseMeters: Float,
)

/** Fits raw depth to metric distance from repeated observations of a known-size QR marker. */
class MarkerAutoCalibrator(
    private val distanceBinMeters: Float = 0.25f,
    private val maxSamplesPerBin: Int = 12,
) {
    private val bins = sortedMapOf<Int, MutableList<CalibrationPair>>()
    private var status = "Arahkan marker ke kamera dan gerakkan melewati beberapa jarak."

    @Synchronized
    fun statusText(): String = status

    @Synchronized
    fun observe(actualMeters: Float, modelMeters: Float): MarkerCalibrationFit? {
        if (!actualMeters.isFinite() || actualMeters !in 0.25f..8f) return null
        if (!modelMeters.isFinite() || modelMeters <= 0f) return null

        val bin = floor(actualMeters / distanceBinMeters).toInt()
        val values = bins.getOrPut(bin) { mutableListOf() }
        values += CalibrationPair(actualMeters, modelMeters)
        while (values.size > maxSamplesPerBin) values.removeAt(0)

        val pairs = bins.values.mapNotNull { samples ->
            if (samples.isEmpty()) null else CalibrationPair(
                actualMeters = samples.map { it.actualMeters }.average().toFloat(),
                modelMeters = samples.map { it.modelMeters }.average().toFloat(),
            )
        }
        if (pairs.size < MIN_DISTANCE_BINS) {
            status = "Butuh $MIN_DISTANCE_BINS jarak berbeda; sekarang ${pairs.size}."
            return null
        }

        val actualRange = pairs.maxOf { it.actualMeters } - pairs.minOf { it.actualMeters }
        val modelRange = pairs.maxOf { it.modelMeters } - pairs.minOf { it.modelMeters }
        if (actualRange < MIN_ACTUAL_RANGE_METERS || modelRange < MIN_MODEL_RANGE_METERS) {
            status = String.format(
                java.util.Locale.US,
                "Rentang kurang: asli %.2f m, depth %.2f m.",
                actualRange,
                modelRange,
            )
            return null
        }

        val fitted = runCatching { LinearDepthCalibration.fit(pairs) }.getOrNull()
        if (fitted == null) {
            status = "Pasangan depth belum cukup bervariasi untuk di-fit."
            return null
        }
        val meanActual = pairs.map { it.actualMeters.toDouble() }.average()
        val total = pairs.sumOf { (it.actualMeters - meanActual) * (it.actualMeters - meanActual) }
        val residual = pairs.sumOf {
            val error = it.actualMeters - fitted.apply(it.modelMeters)
            (error * error).toDouble()
        }
        if (total <= 0.0) {
            status = "Jarak asli belum cukup bervariasi."
            return null
        }

        val rSquared = (1.0 - residual / total).toFloat()
        val rmse = sqrt(residual / pairs.size).toFloat()
        if (rSquared < MIN_R_SQUARED || rmse > MAX_RMSE_METERS) {
            status = String.format(
                java.util.Locale.US,
                "Fit belum lolos: R2 %.2f (min %.2f), galat %.2f m (maks %.2f m).",
                rSquared,
                MIN_R_SQUARED,
                rmse,
                MAX_RMSE_METERS,
            )
            return null
        }

        status = "Fit lolos; kalibrasi otomatis disimpan."
        return MarkerCalibrationFit(fitted, pairs, rSquared, rmse)
    }

    @Synchronized
    fun reset() {
        bins.clear()
        status = "Arahkan marker ke kamera dan gerakkan melewati beberapa jarak."
    }

    @Synchronized
    fun distanceBinCount(): Int = bins.size

    companion object {
        private const val MIN_DISTANCE_BINS = 5
        private const val MIN_ACTUAL_RANGE_METERS = 0.75f
        private const val MIN_MODEL_RANGE_METERS = 0.15f
        private const val MIN_R_SQUARED = 0.80f
        private const val MAX_RMSE_METERS = 0.45f
    }
}
