package id.ac.comvis.yolo26depth

import java.util.ArrayDeque
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

data class CalibrationPair(
    val actualMeters: Float,
    val modelMeters: Float,
)

data class LinearDepthCalibration(
    val a: Float = 1f,
    val b: Float = 0f,
) {
    init {
        require(a.isFinite() && a > 0f)
        require(b.isFinite())
    }

    fun apply(modelMeters: Float): Float = a * modelMeters + b

    companion object {
        val IDENTITY = LinearDepthCalibration()

        fun fit(pairs: List<CalibrationPair>): LinearDepthCalibration {
            // The trained target is depth_m/10, not disparity, so fit the direct distance values.
            require(pairs.size >= 2) { "Masukkan sedikitnya dua pasangan kalibrasi." }
            require(pairs.all {
                it.actualMeters.isFinite() && it.actualMeters > 0f &&
                    it.modelMeters.isFinite() && it.modelMeters > 0f
            }) {
                "Semua jarak kalibrasi harus angka positif dalam meter."
            }
            val meanModel = pairs.map { it.modelMeters.toDouble() }.average()
            val meanActual = pairs.map { it.actualMeters.toDouble() }.average()
            var covariance = 0.0
            var variance = 0.0
            for (pair in pairs) {
                val dx = pair.modelMeters - meanModel
                covariance += dx * (pair.actualMeters - meanActual)
                variance += dx * dx
            }
            require(variance > 1e-9) {
                "Nilai jarak model harus bervariasi; pasangan semua sama tidak dapat di-fit."
            }
            val slope = covariance / variance
            val intercept = meanActual - slope * meanModel
            require(slope.isFinite() && slope > 0.0 && intercept.isFinite()) {
                "Hasil regresi tidak valid. Periksa kembali pasangan jarak."
            }
            return LinearDepthCalibration(slope.toFloat(), intercept.toFloat())
        }

        fun parsePairs(text: String): List<CalibrationPair> =
            text.split(';', '\n', '\r')
                .map(String::trim)
                .filter(String::isNotEmpty)
                .mapIndexed { index, row ->
                    val values = row.split(',')
                    require(values.size == 2) {
                        "Baris ${index + 1}: gunakan format jarak_asli_m,jarak_model_m."
                    }
                    val actual = values[0].trim().toFloatOrNull()
                    val model = values[1].trim().toFloatOrNull()
                    require(actual != null && model != null &&
                        actual.isFinite() && model.isFinite() && actual > 0f && model > 0f
                    ) {
                        "Baris ${index + 1}: kedua jarak harus angka positif dalam meter."
                    }
                    CalibrationPair(actual, model)
                }

        fun serializePairs(pairs: List<CalibrationPair>): String =
            pairs.joinToString(separator = "\n") {
                String.format(Locale.US, "%.6f,%.6f", it.actualMeters, it.modelMeters)
            }
    }
}

class DepthFrameHistory(private val maxFrames: Int = 7) {
    private data class DepthFrame(
        val width: Int,
        val height: Int,
        val values: FloatArray,
    )

    private val frames = ArrayDeque<DepthFrame>()

    init {
        require(maxFrames in 5..10)
    }

    @Synchronized
    fun add(values: FloatArray, width: Int, height: Int) {
        require(width > 0 && height > 0 && values.size == width * height)
        frames.addLast(DepthFrame(width, height, values.copyOf()))
        while (frames.size > maxFrames) frames.removeFirst()
    }

    @Synchronized
    fun sampleMedian(x: Float, y: Float, patchSize: Int): Float? {
        val samples = frames.mapNotNull { frame ->
            DepthMath.sampleMedian(frame.values, frame.width, frame.height, x, y, patchSize)
        }.sorted()
        if (samples.isEmpty()) return null
        return samples[samples.size / 2]
    }
}

class DetectionDistanceSmoother(
    private val maxFrames: Int = 7,
    private val minimumIou: Float = 0.2f,
) {
    private data class Track(
        val id: Int,
        val classId: Int,
        var left: Float,
        var top: Float,
        var right: Float,
        var bottom: Float,
        var missedFrames: Int = 0,
        val rawDistances: ArrayDeque<Float> = ArrayDeque(),
    )

    private val tracks = mutableListOf<Track>()
    private var nextTrackId = 1

    init {
        require(maxFrames in 5..10)
        require(minimumIou in 0f..1f)
    }

    @Synchronized
    fun smooth(
        detections: List<Detection>,
        calibration: LinearDepthCalibration,
    ): List<Detection> {
        tracks.forEach { it.missedFrames++ }
        val usedTrackIds = mutableSetOf<Int>()
        val smoothed = detections.map { detection ->
            val match = tracks
                .asSequence()
                .filter { it.classId == detection.classId && it.id !in usedTrackIds }
                .map { it to intersectionOverUnion(it, detection) }
                .filter { (_, iou) -> iou >= minimumIou }
                .maxByOrNull { (_, iou) -> iou }
                ?.first
                ?: Track(
                    id = nextTrackId++,
                    classId = detection.classId,
                    left = detection.left,
                    top = detection.top,
                    right = detection.right,
                    bottom = detection.bottom,
                ).also(tracks::add)

            usedTrackIds += match.id
            match.left = detection.left
            match.top = detection.top
            match.right = detection.right
            match.bottom = detection.bottom
            match.missedFrames = 0
            detection.rawDistanceMeters
                ?.takeIf { it.isFinite() && it > 0f }
                ?.let(match.rawDistances::addLast)
            while (match.rawDistances.size > maxFrames) match.rawDistances.removeFirst()

            val rawMedian = match.rawDistances
                .takeIf { it.isNotEmpty() }
                ?.sorted()
                ?.let { it[it.size / 2] }
            detection.copy(
                rawDistanceMeters = rawMedian,
                distanceMeters = rawMedian?.let(calibration::apply),
            )
        }
        tracks.removeAll { it.missedFrames > maxFrames }
        return smoothed
    }

    @Synchronized
    fun reset() {
        tracks.clear()
        nextTrackId = 1
    }

    private fun intersectionOverUnion(track: Track, detection: Detection): Float {
        val intersectionLeft = max(track.left, detection.left)
        val intersectionTop = max(track.top, detection.top)
        val intersectionRight = min(track.right, detection.right)
        val intersectionBottom = min(track.bottom, detection.bottom)
        val intersection = (intersectionRight - intersectionLeft).coerceAtLeast(0f) *
            (intersectionBottom - intersectionTop).coerceAtLeast(0f)
        val trackArea = (track.right - track.left) * (track.bottom - track.top)
        val detectionArea = (detection.right - detection.left) *
            (detection.bottom - detection.top)
        return intersection / (trackArea + detectionArea - intersection).coerceAtLeast(1e-6f)
    }
}
