package id.ac.comvis.yolo26depth

data class Detection(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val distanceMeters: Float?,
)

data class FrameResult(
    val width: Int,
    val height: Int,
    val detections: List<Detection>,
    val metricDepth: FloatArray,
    val depthWidth: Int,
    val depthHeight: Int,
    val depthColor: IntArray,
    val inferenceLatencyMs: Float,
    val fps: Float,
    val runtimeLabel: String,
)

data class LetterboxTransform(
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val originalWidth: Int,
    val originalHeight: Int,
)

data class ModelBox(
    val classId: Int,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
)
