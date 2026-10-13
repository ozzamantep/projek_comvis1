package id.ac.comvis.yolo26depth

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import android.graphics.pdf.PdfDocument
import android.hardware.camera2.CameraCharacteristics
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.Settings
import android.view.Gravity
import android.view.MotionEvent
import android.view.Surface
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.barcode.BarcodeScanner
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.google.zxing.BarcodeFormat
import com.google.zxing.EncodeHintType
import com.google.zxing.qrcode.QRCodeWriter
import com.google.zxing.qrcode.decoder.ErrorCorrectionLevel
import java.io.Closeable
import java.io.IOException
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt
import org.json.JSONObject

// CalibrationMarker
object CalibrationMarker {
    const val PAYLOAD = "CVDEPTH10"
    const val SIDE_METERS = 0.10f
    private const val MODULES = 21
    private const val QUIET_ZONE_MM = 20.0
    private const val QR_SIDE_MM = 100.0
    private const val PAGE_SIDE_MM = QUIET_ZONE_MM * 2 + QR_SIDE_MM

    fun writePdf(output: OutputStream) {
        val matrix = QRCodeWriter().encode(
            PAYLOAD,
            BarcodeFormat.QR_CODE,
            MODULES,
            MODULES,
            mapOf(
                EncodeHintType.MARGIN to 0,
                EncodeHintType.ERROR_CORRECTION to ErrorCorrectionLevel.M,
                EncodeHintType.CHARACTER_SET to "UTF-8",
            ),
        )
        require(matrix.width == MODULES && matrix.height == MODULES) {
            "Marker QR tidak terbentuk dalam ukuran yang diharapkan."
        }

        val pagePoints = millimetersToPoints(PAGE_SIDE_MM).toInt()
        val quietZonePoints = millimetersToPoints(QUIET_ZONE_MM)
        val modulePoints = millimetersToPoints(QR_SIDE_MM) / MODULES
        val document = PdfDocument()
        try {
            val page = document.startPage(
                PdfDocument.PageInfo.Builder(pagePoints, pagePoints, 1).create(),
            )
            val paint = Paint().apply {
                color = Color.BLACK
                style = Paint.Style.FILL
                isAntiAlias = false
            }
            for (y in 0 until matrix.height) {
                for (x in 0 until matrix.width) {
                    if (matrix[x, y]) {
                        val left = quietZonePoints + x * modulePoints
                        val top = quietZonePoints + y * modulePoints
                        page.canvas.drawRect(
                            RectF(
                                left.toFloat(),
                                top.toFloat(),
                                (left + modulePoints).toFloat(),
                                (top + modulePoints).toFloat(),
                            ),
                            paint,
                        )
                    }
                }
            }
            document.finishPage(page)
            document.writeTo(output)
            output.flush()
        } finally {
            document.close()
        }
    }

    private fun millimetersToPoints(mm: Double): Double = mm * 72.0 / 25.4
}

// CameraFrameConverter
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

// CroppedDepthMap
data class CroppedDepthMap(
    val values: FloatArray,
    val width: Int,
    val height: Int,
)

// DepthCalibration
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

// DepthMapView
class DepthMapView(context: Context) : View(context) {
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 255, 255)
        style = Paint.Style.STROKE
        strokeWidth = 2f * resources.displayMetrics.density
    }
    private val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(210, 0, 0, 0)
        style = Paint.Style.FILL
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 10f * resources.displayMetrics.density
        setShadowLayer(1f, 0f, 0f, Color.BLACK)
    }
    private var bitmap: Bitmap? = null
    private var frameWidth = 0
    private var frameHeight = 0
    private var detections: List<Detection> = emptyList()

    fun update(result: FrameResult?, value: Bitmap?) {
        bitmap = value
        frameWidth = result?.width ?: 0
        frameHeight = result?.height ?: 0
        detections = result?.detections.orEmpty()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(0xff101010.toInt())
        val source = bitmap ?: return
        if (source.isRecycled) return
        val imageRect = fittedImageRect(source)
        canvas.drawBitmap(source, null, imageRect, paint)
        if (frameWidth <= 0 || frameHeight <= 0) return

        val scaleX = imageRect.width() / frameWidth
        val scaleY = imageRect.height() / frameHeight
        for (detection in detections) {
            val box = RectF(
                imageRect.left + detection.left * scaleX,
                imageRect.top + detection.top * scaleY,
                imageRect.left + detection.right * scaleX,
                imageRect.top + detection.bottom * scaleY,
            )
            canvas.drawRect(box, boxPaint)

            val distance = detection.distanceMeters?.let {
                String.format(Locale.US, "%.2f m", it)
            } ?: "jarak -"
            val label = "${detection.className} · $distance"
            val maxLabelWidth = imageRect.width().coerceAtLeast(1f)
            val oldTextSize = labelPaint.textSize
            val textWidth = labelPaint.measureText(label)
            if (textWidth > maxLabelWidth) {
                labelPaint.textSize = oldTextSize * maxLabelWidth / textWidth
            }
            val labelWidth = labelPaint.measureText(label) + 8f
            val labelHeight = labelPaint.textSize + 8f
            val labelLeft = box.left.coerceIn(
                imageRect.left,
                (imageRect.right - labelWidth).coerceAtLeast(imageRect.left),
            )
            val labelTop = if (box.top - labelHeight >= imageRect.top) {
                box.top - labelHeight
            } else {
                box.top.coerceAtMost(imageRect.bottom - labelHeight)
            }
            canvas.drawRect(
                RectF(labelLeft, labelTop, labelLeft + labelWidth, labelTop + labelHeight),
                labelBackgroundPaint,
            )
            canvas.drawText(
                label,
                labelLeft + 4f,
                labelTop + labelPaint.textSize + 2f,
                labelPaint,
            )
            labelPaint.textSize = oldTextSize
        }
    }

    private fun fittedImageRect(source: Bitmap): RectF {
        val aspectWidth = if (frameWidth > 0) frameWidth else source.width
        val aspectHeight = if (frameHeight > 0) frameHeight else source.height
        val scale = minOf(width.toFloat() / aspectWidth, height.toFloat() / aspectHeight)
        val drawnWidth = aspectWidth * scale
        val drawnHeight = aspectHeight * scale
        val left = (width - drawnWidth) / 2f
        val top = (height - drawnHeight) / 2f
        return RectF(left, top, left + drawnWidth, top + drawnHeight)
    }
}

// DepthMath
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

// Detection
data class Detection(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val rawDistanceMeters: Float?,
    val distanceMeters: Float?,
    val rawCenterX: Float,
    val rawCenterY: Float,
    val rawWidth: Float,
    val rawHeight: Float,
)

data class DepthMeasurement(
    val modelMeters: Float,
    val calibratedMeters: Float,
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
    val detectorMaxScore: Float,
    val detectorCandidatesAboveThreshold: Int,
    val detectorValidBeforeNms: Int,
    val detectorCoordinatesNormalized: Boolean,
)

data class DetectionDecodeResult(
    val boxes: List<ModelBox>,
    val maxScore: Float,
    val candidatesAboveThreshold: Int,
    val validBoxesBeforeNms: Int,
    val coordinatesNormalized: Boolean,
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
    val rawCenterX: Float,
    val rawCenterY: Float,
    val rawWidth: Float,
    val rawHeight: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
)

// DetectionOverlayView
class DetectionOverlayView(context: Context) : View(context) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(255, 193, 7)
        style = Paint.Style.STROKE
        strokeWidth = 3.5f * resources.displayMetrics.density
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(88, 20, 20, 20)
        style = Paint.Style.FILL
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 13f * resources.displayMetrics.density
        style = Paint.Style.FILL
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }
    private val depthPaint = Paint(Paint.FILTER_BITMAP_FLAG).apply { alpha = 92 }
    private val tapPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 3f * resources.displayMetrics.density
    }

    private var frameWidth = 0
    private var frameHeight = 0
    private var detections: List<Detection> = emptyList()
    private var depthBitmap: Bitmap? = null
    private var showDepth = false
    private var tapPoint: Pair<Float, Float>? = null
    private var tapModelDistance: Float? = null
    private var tapDistance: Float? = null
    var onMeasureTap: ((Float, Float) -> Unit)? = null

    init {
        isClickable = true
    }

    fun update(
        result: FrameResult,
        bitmap: Bitmap?,
        depthVisible: Boolean,
        measurePoint: Pair<Float, Float>?,
        measureModelMeters: Float?,
        measureMeters: Float?,
    ) {
        frameWidth = result.width
        frameHeight = result.height
        detections = result.detections
        depthBitmap = bitmap
        showDepth = depthVisible
        tapPoint = measurePoint
        tapModelDistance = measureModelMeters
        tapDistance = measureMeters
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (frameWidth <= 0 || frameHeight <= 0) return
        val imageRect = fittedImageRect()
        val depth = depthBitmap
        if (showDepth && depth != null && !depth.isRecycled) {
            canvas.drawBitmap(depth, null, imageRect, depthPaint)
        }

        val scaleX = imageRect.width() / frameWidth
        val scaleY = imageRect.height() / frameHeight
        for (detection in detections) {
            val rect = RectF(
                imageRect.left + detection.left * scaleX,
                imageRect.top + detection.top * scaleY,
                imageRect.left + detection.right * scaleX,
                imageRect.top + detection.bottom * scaleY,
            )
            canvas.drawRect(rect, boxPaint)
            val distance = detection.distanceMeters?.let {
                String.format(Locale.US, "%.2f m / %.0f cm", it, it * 100f)
            } ?: "jarak tidak tersedia"
            val label = "${detection.className} - $distance"
            val textWidth = labelPaint.measureText(label)
            val labelHeight = labelPaint.textSize + 10f
            val labelTop = (rect.top - labelHeight).coerceAtLeast(imageRect.top)
            val labelRect = RectF(
                rect.left,
                labelTop,
                (rect.left + textWidth + 12f).coerceAtMost(imageRect.right),
                labelTop + labelHeight,
            )
            canvas.drawRect(labelRect, fillPaint)
            canvas.drawText(
                label,
                labelRect.left + 6f,
                labelRect.bottom - 5f,
                labelPaint,
            )
        }

        tapPoint?.let { point ->
            val x = imageRect.left + point.first * scaleX
            val y = imageRect.top + point.second * scaleY
            canvas.drawCircle(x, y, 12f * resources.displayMetrics.density, tapPaint)
            tapDistance?.let { distance ->
                val modelDistance = tapModelDistance
                canvas.drawText(
                    String.format(
                        Locale.US,
                        "Tap: %.2f m (model %.2f m)",
                        distance,
                        modelDistance ?: distance,
                    ),
                    x + 14f,
                    y - 14f,
                    labelPaint,
                )
            }
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_UP || frameWidth <= 0 || frameHeight <= 0) {
            return true
        }
        val rect = fittedImageRect()
        if (!rect.contains(event.x, event.y)) return true
        val x = ((event.x - rect.left) * frameWidth / rect.width())
            .coerceIn(0f, frameWidth.toFloat())
        val y = ((event.y - rect.top) * frameHeight / rect.height())
            .coerceIn(0f, frameHeight.toFloat())
        onMeasureTap?.invoke(x, y)
        performClick()
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun fittedImageRect(): RectF {
        val scale = minOf(width.toFloat() / frameWidth, height.toFloat() / frameHeight)
        val fittedWidth = frameWidth * scale
        val fittedHeight = frameHeight * scale
        val left = (width - fittedWidth) / 2f
        val top = (height - fittedHeight) / 2f
        return RectF(left, top, left + fittedWidth, top + fittedHeight)
    }

    fun mapFrameBoxToView(detection: Detection): RectF? {
        if (frameWidth <= 0 || frameHeight <= 0 || width <= 0 || height <= 0) return null
        val imageRect = fittedImageRect()
        val scaleX = imageRect.width() / frameWidth
        val scaleY = imageRect.height() / frameHeight
        return RectF(
            imageRect.left + detection.left * scaleX,
            imageRect.top + detection.top * scaleY,
            imageRect.left + detection.right * scaleX,
            imageRect.top + detection.bottom * scaleY,
        )
    }
}

// ImagePreprocessor
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

// InferenceEngine
class InferenceEngine(
    context: Context,
    private val config: ModelConfig,
) : Closeable {
    private val appContext = context.applicationContext
    private val ortEnvironment = OrtEnvironment.getEnvironment()
    private val session = createSession()
    private val ema = DepthEma(config.emaAlpha)
    private val depthHistory = DepthFrameHistory(maxFrames = 7)
    private val distanceSmoother = DetectionDistanceSmoother(maxFrames = 7)

    val runtimeLabel: String = "YOLO26n multitask · ONNX Runtime CPU"

    init {
        val inputInfo = session.inputInfo[config.inputName]?.info as? TensorInfo
            ?: error("Input ONNX '${config.inputName}' tidak ditemukan atau bukan tensor.")
        require(inputInfo.shape.contentEquals(longArrayOf(1, 3, config.inputHeight.toLong(), config.inputWidth.toLong()))) {
            "Bentuk input ONNX ${inputInfo.shape.contentToString()} tidak cocok dengan " +
                "[1, 3, ${config.inputHeight}, ${config.inputWidth}]."
        }
        val detectionInfo = session.outputInfo[config.detectionOutputName]?.info as? TensorInfo
            ?: error("Output deteksi ONNX '${config.detectionOutputName}' tidak ditemukan.")
        require(detectionInfo.shape.contentEquals(longArrayOf(1, config.detectorMaxDetections.toLong(), 6))) {
            "Bentuk output deteksi ${detectionInfo.shape.contentToString()} tidak cocok " +
                "dengan [1, ${config.detectorMaxDetections}, 6]."
        }
        val depthInfo = session.outputInfo[config.depthOutputName]?.info as? TensorInfo
            ?: error("Output depth ONNX '${config.depthOutputName}' tidak ditemukan.")
        require(depthInfo.shape.contentEquals(longArrayOf(1, 1, config.inputHeight.toLong(), config.inputWidth.toLong()))) {
            "Bentuk output depth ${depthInfo.shape.contentToString()} tidak cocok dengan " +
                "[1, 1, ${config.inputHeight}, ${config.inputWidth}]."
        }
    }

    fun runFrame(
        bitmap: Bitmap,
        confidenceThreshold: Float,
        selectedClasses: Set<String>,
        calibration: LinearDepthCalibration,
        fps: Float,
    ): FrameResult {
        val (inputBuffer, transform) = ImagePreprocessor.multitaskInput(
            bitmap,
            config.inputWidth,
            config.inputHeight,
        )
        val inferenceStart = SystemClock.elapsedRealtimeNanos()
        val tensor = OnnxTensor.createTensor(
            ortEnvironment,
            inputBuffer.asFloatBuffer(),
            longArrayOf(1, 3, config.inputHeight.toLong(), config.inputWidth.toLong()),
        )

        val (detectionOutput, depthRaw) = tensor.use { inputTensor ->
            session.run(mapOf(config.inputName to inputTensor)).use { outputs ->
                val detections = outputs.get(config.detectionOutputName).orElseThrow {
                    IllegalStateException("Output '${config.detectionOutputName}' hilang dari model.")
                } as OnnxTensor
                val depth = outputs.get(config.depthOutputName).orElseThrow {
                    IllegalStateException("Output '${config.depthOutputName}' hilang dari model.")
                } as OnnxTensor
                readTensor(detections, config.detectorMaxDetections * 6) to
                    readTensor(depth, config.inputWidth * config.inputHeight)
            }
        }
        val inferenceLatencyMs = (SystemClock.elapsedRealtimeNanos() - inferenceStart) / 1_000_000f

        val decodeResult = YoloPostprocessor.decodeEndToEnd(
            output = detectionOutput,
            config = config,
            transform = transform,
            confidenceThreshold = confidenceThreshold,
            selectedClasses = selectedClasses,
        )
        val metricDepthSquare = FloatArray(depthRaw.size) { index ->
            depthRaw[index] * config.depthScaleMeters
        }
        val croppedDepth = DepthMath.removeLetterbox(
            metricDepthSquare,
            config.inputWidth,
            config.inputHeight,
            transform,
        )
        val croppedMetricDepth = ema.update(
            croppedDepth.values,
            croppedDepth.width,
            croppedDepth.height,
        )
        depthHistory.add(croppedMetricDepth, croppedDepth.width, croppedDepth.height)

        val detections = decodeResult.boxes.map { box ->
            val scaleX = croppedDepth.width / bitmap.width.toFloat()
            val scaleY = croppedDepth.height / bitmap.height.toFloat()
            val sampled = DepthMath.sampleMedianInBox(
                croppedMetricDepth,
                croppedDepth.width,
                croppedDepth.height,
                box.left * scaleX,
                box.top * scaleY,
                box.right * scaleX,
                box.bottom * scaleY,
            )
            Detection(
                classId = box.classId,
                className = config.classNames[box.classId],
                confidence = box.confidence,
                left = box.left,
                top = box.top,
                right = box.right,
                bottom = box.bottom,
                rawDistanceMeters = sampled,
                distanceMeters = sampled?.let(calibration::apply),
                rawCenterX = box.rawCenterX,
                rawCenterY = box.rawCenterY,
                rawWidth = box.rawWidth,
                rawHeight = box.rawHeight,
            )
        }
        val smoothedDetections = distanceSmoother.smooth(detections, calibration)

        return FrameResult(
            width = bitmap.width,
            height = bitmap.height,
            detections = smoothedDetections,
            metricDepth = croppedMetricDepth,
            depthWidth = croppedDepth.width,
            depthHeight = croppedDepth.height,
            depthColor = InfernoPalette.colorizeForDisplay(croppedMetricDepth),
            inferenceLatencyMs = inferenceLatencyMs,
            fps = fps,
            runtimeLabel = runtimeLabel,
            detectorMaxScore = decodeResult.maxScore,
            detectorCandidatesAboveThreshold = decodeResult.candidatesAboveThreshold,
            detectorValidBeforeNms = decodeResult.validBoxesBeforeNms,
            detectorCoordinatesNormalized = decodeResult.coordinatesNormalized,
        )
    }

    fun sampleTap(
        result: FrameResult,
        frameX: Float,
        frameY: Float,
        calibration: LinearDepthCalibration,
    ): DepthMeasurement? {
        val x = frameX * result.depthWidth / result.width
        val y = frameY * result.depthHeight / result.height
        val modelMeters = depthHistory.sampleMedian(x, y, config.depthPatchSize)
            ?: return null
        return DepthMeasurement(
            modelMeters = modelMeters,
            calibratedMeters = calibration.apply(modelMeters),
        )
    }

    fun resetDistanceSmoothing() {
        distanceSmoother.reset()
    }

    private fun readTensor(tensor: OnnxTensor, count: Int): FloatArray {
        val floatBuffer = tensor.floatBuffer
            ?: error("Output ONNX harus berupa tensor float.")
        require(floatBuffer.remaining() == count) {
            "Jumlah nilai output ONNX ${floatBuffer.remaining()}, seharusnya $count."
        }
        return FloatArray(count).also(floatBuffer::get)
    }

    private fun createSession(): OrtSession {
        val modelBytes = appContext.assets.open(config.model).use { it.readBytes() }
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 4))
        return try {
            ortEnvironment.createSession(modelBytes, options)
        } finally {
            options.close()
        }
    }

    override fun close() {
        session.close()
    }
}

// InfernoPalette
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

// MarkerAutoCalibrator
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

// ModelConfig
data class ModelConfig(
    val model: String,
    val inputName: String,
    val detectionOutputName: String,
    val depthOutputName: String,
    val inputWidth: Int,
    val inputHeight: Int,
    val depthScaleMeters: Float,
    val depthPatchSize: Int,
    val emaAlpha: Float,
    val classNames: List<String>,
    val detectorConfidence: Float,
    val detectorMaxDetections: Int,
) {
    init {
        require(inputWidth > 0 && inputHeight > 0)
        require(depthScaleMeters > 0f)
        require(depthPatchSize > 0 && depthPatchSize % 2 == 1)
        require(emaAlpha > 0f && emaAlpha <= 1f)
        require(classNames.isNotEmpty() && classNames.distinct().size == classNames.size)
        require(detectorConfidence in 0f..1f)
        require(detectorMaxDetections > 0)
    }

    companion object {
        fun load(context: Context): ModelConfig {
            val json = context.assets.open("model_config.json")
                .bufferedReader()
                .use { JSONObject(it.readText()) }
            val names = json.getJSONArray("class_names")
            return ModelConfig(
                model = json.getString("model"),
                inputName = json.getString("input_name"),
                detectionOutputName = json.getString("detection_output_name"),
                depthOutputName = json.getString("depth_output_name"),
                inputWidth = json.getInt("input_width"),
                inputHeight = json.getInt("input_height"),
                depthScaleMeters = json.getDouble("depth_scale_m").toFloat(),
                depthPatchSize = json.getInt("depth_patch_size"),
                emaAlpha = json.getDouble("ema_alpha").toFloat(),
                classNames = List(names.length()) { names.getString(it) },
                detectorConfidence = json.getDouble("detector_confidence").toFloat(),
                detectorMaxDetections = json.getInt("detector_max_detections"),
            )
        }
    }
}

// YoloPostprocessor
object YoloPostprocessor {
    /** Decode ONNX output [1, maxDetections, 6]: xyxy, confidence, class index. */
    fun decodeEndToEnd(
        output: FloatArray,
        config: ModelConfig,
        transform: LetterboxTransform,
        confidenceThreshold: Float,
        selectedClasses: Set<String>,
    ): DetectionDecodeResult {
        val rowSize = 6
        require(output.size == config.detectorMaxDetections * rowSize) {
            "Detection output length ${output.size}; expected " +
                "${config.detectorMaxDetections * rowSize}."
        }

        val maxCoordinate = (0 until config.detectorMaxDetections).maxOf { row ->
            val offset = row * rowSize
            maxOf(
                kotlin.math.abs(output[offset]),
                kotlin.math.abs(output[offset + 1]),
                kotlin.math.abs(output[offset + 2]),
                kotlin.math.abs(output[offset + 3]),
            )
        }
        val coordinatesNormalized = maxCoordinate <= 2f && maxCoordinate > 0f
        val scaleX = if (coordinatesNormalized) config.inputWidth.toFloat() else 1f
        val scaleY = if (coordinatesNormalized) config.inputHeight.toFloat() else 1f
        val boxes = ArrayList<ModelBox>()
        var maxScore = 0f
        var candidatesAboveThreshold = 0
        var validRows = 0

        for (row in 0 until config.detectorMaxDetections) {
            val offset = row * rowSize
            val score = output[offset + 4]
            val classValue = output[offset + 5]
            if (!score.isFinite() || !classValue.isFinite()) continue
            if (score <= 0f || classValue < 0f) continue

            val classId = classValue.roundToInt()
            if (classId !in config.classNames.indices) continue
            val x1 = output[offset] * scaleX
            val y1 = output[offset + 1] * scaleY
            val x2 = output[offset + 2] * scaleX
            val y2 = output[offset + 3] * scaleY
            if (!x1.isFinite() || !y1.isFinite() || !x2.isFinite() || !y2.isFinite()) continue
            if (x2 <= x1 || y2 <= y1) continue

            validRows++
            maxScore = maxOf(maxScore, score)
            if (score < confidenceThreshold) continue
            candidatesAboveThreshold++

            val className = config.classNames[classId]
            if (selectedClasses.isNotEmpty() && className.lowercase() !in selectedClasses) continue

            val left = ((x1 - transform.padX) / transform.scale)
                .coerceIn(0f, transform.originalWidth.toFloat())
            val top = ((y1 - transform.padY) / transform.scale)
                .coerceIn(0f, transform.originalHeight.toFloat())
            val right = ((x2 - transform.padX) / transform.scale)
                .coerceIn(0f, transform.originalWidth.toFloat())
            val bottom = ((y2 - transform.padY) / transform.scale)
                .coerceIn(0f, transform.originalHeight.toFloat())
            if (right <= left || bottom <= top) continue

            boxes += ModelBox(
                classId = classId,
                confidence = score,
                rawCenterX = (x1 + x2) / 2f,
                rawCenterY = (y1 + y2) / 2f,
                rawWidth = x2 - x1,
                rawHeight = y2 - y1,
                left = left,
                top = top,
                right = right,
                bottom = bottom,
            )
        }

        return DetectionDecodeResult(
            boxes = boxes,
            maxScore = maxScore,
            candidatesAboveThreshold = candidatesAboveThreshold,
            validBoxesBeforeNms = validRows,
            coordinatesNormalized = coordinatesNormalized,
        )
    }
}

// MainActivity
class MainActivity : androidx.activity.ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: DetectionOverlayView
    private lateinit var depthMapView: DepthMapView
    private lateinit var statusView: TextView
    private lateinit var calibrationSummaryView: TextView
    private lateinit var warningView: TextView
    private lateinit var permissionButton: Button
    private lateinit var frameRow: LinearLayout
    private lateinit var previewHost: android.widget.FrameLayout
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var markerExecutor: ExecutorService
    private lateinit var markerScanner: BarcodeScanner
    private lateinit var config: ModelConfig
    private lateinit var calibrationPairsInput: EditText
    private var inferenceEngine: InferenceEngine? = null
    private var latestResult: FrameResult? = null
    private var latestDepthBitmap: Bitmap? = null
    private var tapPoint: Pair<Float, Float>? = null
    private var tapMeasurement: DepthMeasurement? = null
    private var showDepthOverlay = false
    private var sideBySide = true
    @Volatile private var confidenceThreshold = 0.5f
    @Volatile private var selectedClasses: Set<String> = emptySet()
    @Volatile private var depthCalibration = LinearDepthCalibration.IDENTITY
    private var lastFpsWindowNs = 0L
    private var framesInFpsWindow = 0
    private var displayedFps = 0f
    private val isBindingCamera = AtomicBoolean(false)
    private val markerScanPending = AtomicBoolean(false)
    private val markerAutoCalibrator = MarkerAutoCalibrator()
    @Volatile private var markerCalibrationEnabled = false
    @Volatile private var cameraFocalLengthMm: Float? = null
    @Volatile private var cameraSensorSizeMm: android.util.SizeF? = null
    @Volatile private var lastAutoCalibrationSaveMs = 0L
    @Volatile private var markerOpticsWarningShown = false
    private var permissionPermanentlyDenied = false
    private var startupError: String? = null
    private var cameraBound = false
    private var previewUseCase: Preview? = null
    private var analysisUseCase: ImageAnalysis? = null
    private val calibrationPreferences by lazy {
        getSharedPreferences(CALIBRATION_PREFERENCES, MODE_PRIVATE)
    }

    private val csvExportLauncher =
        registerForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { uri ->
            if (uri != null) exportCalibrationCsv(uri)
        }

    private val markerExportLauncher =
        registerForActivityResult(ActivityResultContracts.CreateDocument("application/pdf")) { uri ->
            if (uri != null) exportCalibrationMarker(uri)
        }

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                permissionPermanentlyDenied = false
                permissionButton.visibility = View.GONE
                statusView.text = "Izin kamera diberikan; membuka kamera..."
                bindCamera()
            } else {
                permissionPermanentlyDenied =
                    !shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)
                statusView.text = if (!permissionPermanentlyDenied) {
                    "Izin kamera ditolak. Izinkan kamera untuk menjalankan deteksi dan depth."
                } else {
                    "Izin kamera ditolak permanen. Buka Pengaturan aplikasi untuk mengizinkan kamera."
                }
                permissionButton.text = if (!permissionPermanentlyDenied) {
                    "Coba izin kamera lagi"
                } else {
                    "Buka pengaturan aplikasi"
                }
                permissionButton.visibility = View.VISIBLE
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        markerExecutor = Executors.newSingleThreadExecutor()
        markerScanner = BarcodeScanning.getClient(
            BarcodeScannerOptions.Builder()
                .setBarcodeFormats(Barcode.FORMAT_QR_CODE)
                .build(),
        )
        try {
            config = ModelConfig.load(this)
            confidenceThreshold = config.detectorConfidence
            if (calibrationPreferences.getInt(CALIBRATION_PROFILE_VERSION, 0) < 3) {
                val saved = calibrationPreferences.edit()
                    .putFloat(CALIBRATION_A, 1f)
                    .putFloat(CALIBRATION_B, 0f)
                    .putString(CALIBRATION_PAIRS, "")
                    .putInt(CALIBRATION_PROFILE_VERSION, 3)
                    .commit()
                if (!saved) throw IOException("Reset kalibrasi ke nilai model mentah gagal disimpan.")
            }
            depthCalibration = LinearDepthCalibration(
                a = calibrationPreferences.getFloat(CALIBRATION_A, 1f),
                b = calibrationPreferences.getFloat(CALIBRATION_B, 0f),
            )
            inferenceEngine = InferenceEngine(this, config)
        } catch (error: Exception) {
            startupError = "Gagal memuat model/config: ${error.message}"
            showFatalError("Gagal memuat model/config: ${error.message}")
        }
        createUi()
        if (inferenceEngine != null) {
            if (hasCameraPermission()) {
                bindCamera()
            } else {
                statusView.text = "Aplikasi memerlukan izin kamera."
                permissionButton.visibility = View.VISIBLE
            }
        } else {
            statusView.text = startupError ?: "Model tidak berhasil disiapkan."
        }
    }

    override fun onResume() {
        super.onResume()
        if (
            ::permissionButton.isInitialized &&
            hasCameraPermission() &&
            !cameraBound
        ) {
            permissionPermanentlyDenied = false
            permissionButton.visibility = View.GONE
            bindCamera()
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        val rotation = previewView.display?.rotation ?: Surface.ROTATION_0
        previewUseCase?.targetRotation = rotation
        analysisUseCase?.targetRotation = rotation
    }

    private fun createUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xff101010.toInt())
        }

        warningView = text(
            "Model depth sudah terkalibrasi dalam meter; kalibrasi khusus kamera tetap disarankan. " +
                "Detector decode/NMS BELUM diverifikasi terhadap Ultralytics PC pada gambar nyata.",
            13f,
        ).apply {
            setTextColor(0xffffd54f.toInt())
            setPadding(dp(10), dp(8), dp(10), dp(8))
        }
        root.addView(
            warningView,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )

        frameRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        previewHost = android.widget.FrameLayout(this)
        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
        }
        overlayView = DetectionOverlayView(this)
        previewHost.addView(
            previewView,
            android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )
        previewHost.addView(
            overlayView,
            android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )
        depthMapView = DepthMapView(this)
        depthMapView.visibility = View.VISIBLE
        frameRow.addView(
            previewHost,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f),
        )
        frameRow.addView(
            depthMapView,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 0f),
        )
        root.addView(
            frameRow,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f,
            ),
        )

        statusView = text("Menyiapkan model...", 13f).apply {
            setTextColor(0xffeeeeee.toInt())
            setPadding(dp(10), dp(6), dp(10), dp(6))
        }
        root.addView(statusView)

        permissionButton = Button(this).apply {
            text = "Izinkan kamera"
            visibility = View.GONE
            setOnClickListener {
                if (permissionPermanentlyDenied) {
                    startActivity(
                        Intent(
                            Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                            android.net.Uri.parse("package:$packageName"),
                        ),
                    )
                } else {
                    permissionLauncher.launch(Manifest.permission.CAMERA)
                }
            }
        }
        root.addView(permissionButton)

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(8), dp(2), dp(8), dp(8))
        }
        val confidenceLabel = text("Confidence: ${confidenceThreshold.format(2)}", 12f)
        val confidenceSeekBar = SeekBar(this).apply {
            max = 100
            progress = (confidenceThreshold / MAX_CONFIDENCE * 100).toInt()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    confidenceThreshold = progress / 100f * MAX_CONFIDENCE
                    confidenceLabel.text = "Confidence: ${confidenceThreshold.format(2)}"
                }

                override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
            })
        }
        controls.addView(confidenceLabel)
        controls.addView(confidenceSeekBar)
        controls.addView(Button(this).apply {
            text = "Reset confidence ke 0.05"
            setOnClickListener {
                confidenceThreshold = DEFAULT_CONFIDENCE
                confidenceSeekBar.progress =
                    (DEFAULT_CONFIDENCE / MAX_CONFIDENCE * 100).toInt()
                confidenceLabel.text = "Confidence: ${confidenceThreshold.format(2)}"
            }
        })

        val filterRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        filterRow.addView(text("Kelas (kosong = semua):", 12f))
        val classFilter = EditText(this).apply {
            setSingleLine(true)
            hint = "chair, curtain, computer"
            setTextColor(0xffffffff.toInt())
            setHintTextColor(0xffaaaaaa.toInt())
            textSize = 13f
            imeOptions = EditorInfo.IME_ACTION_DONE
            setOnEditorActionListener { editor, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_DONE) {
                    selectedClasses = parseClasses((editor as EditText).text.toString())
                    statusView.text = classFilterStatus(selectedClasses)
                    true
                } else {
                    false
                }
            }
        }
        filterRow.addView(classFilter, LinearLayout.LayoutParams(0, dp(48), 1f))
        filterRow.addView(Button(this).apply {
            text = "Pakai"
            setOnClickListener {
                selectedClasses = parseClasses(classFilter.text.toString())
                statusView.text = classFilterStatus(selectedClasses)
                classFilter.clearFocus()
            }
        })
        filterRow.addView(Button(this).apply {
            text = "Semua"
            setOnClickListener {
                classFilter.text.clear()
                selectedClasses = emptySet()
                statusView.text = classFilterStatus(selectedClasses)
                classFilter.clearFocus()
            }
        })
        controls.addView(filterRow)

        controls.addView(text("Kalibrasi linear: jarak_asli_m, jarak_model_m", 12f))
        calibrationPairsInput = EditText(this).apply {
            setSingleLine(false)
            minLines = 3
            maxLines = 5
            gravity = Gravity.TOP or Gravity.START
            setText(calibrationPreferences.getString(CALIBRATION_PAIRS, "").orEmpty())
            hint = "Format: jarak_asli_m,jarak_model_m; satu pasangan per baris"
            setTextColor(0xffffffff.toInt())
            setHintTextColor(0xffaaaaaa.toInt())
            textSize = 14f
            inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE or
                android.text.InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        }
        controls.addView(
            calibrationPairsInput,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(110),
            ),
        )
        val calibrationButtons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        calibrationButtons.addView(Button(this).apply {
            text = "Fit & simpan"
            setOnClickListener { fitAndSaveCalibration() }
        })
        calibrationButtons.addView(Button(this).apply {
            text = "Ekspor CSV"
            setOnClickListener {
                csvExportLauncher.launch("depth_validation.csv")
            }
        })
        calibrationButtons.addView(Button(this).apply {
            text = "Reset ke mentah"
            setOnClickListener { resetCalibrationToRaw() }
        })
        controls.addView(calibrationButtons)
        val markerCalibrationRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        markerCalibrationRow.addView(Switch(this).apply {
            text = "Auto kalibrasi QR 10 cm"
            setTextColor(0xffffffff.toInt())
            setOnCheckedChangeListener { _, enabled ->
                markerCalibrationEnabled = enabled
                if (enabled) {
                    markerAutoCalibrator.reset()
                    statusView.text =
                        "Arahkan QR 10 cm ke kamera; gerakkan perlahan dari 0,5 sampai 3 m."
                } else {
                    statusView.text = "Kalibrasi marker otomatis dijeda."
                }
            }
        })
        markerCalibrationRow.addView(Button(this).apply {
            text = "Simpan marker PDF"
            setOnClickListener { markerExportLauncher.launch("marker_kalibrasi_10cm.pdf") }
        })
        controls.addView(markerCalibrationRow)
        controls.addView(
            text("Cetak PDF ukuran sebenarnya (kotak hitam 10 × 10 cm); hadapkan ke kamera.", 11f),
        )
        calibrationSummaryView = text(calibrationDescription(), 12f)
        controls.addView(calibrationSummaryView)

        val switchRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val depthSwitch = Switch(this).apply {
            text = "Depth overlay"
            setTextColor(0xffffffff.toInt())
            setOnCheckedChangeListener { _, checked ->
                showDepthOverlay = checked
                redrawLatest()
            }
        }
        val sideBySideSwitch = Switch(this).apply {
            text = "Side-by-side"
            setTextColor(0xffffffff.toInt())
            isChecked = sideBySide
            setOnCheckedChangeListener { _, checked ->
                sideBySide = checked
                depthMapView.visibility = if (checked) View.VISIBLE else View.GONE
                updateDisplayWeights()
                redrawLatest()
            }
        }
        switchRow.addView(depthSwitch)
        switchRow.addView(sideBySideSwitch)
        controls.addView(switchRow)

        val scroll = ScrollView(this).apply {
            isFillViewport = false
            addView(controls)
        }
        root.addView(
            scroll,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )
        setContentView(root)
        updateDisplayWeights()

        overlayView.onMeasureTap = { x, y ->
            val result = latestResult
            val engine = inferenceEngine
            if (result == null || engine == null) {
                statusView.text = "Belum ada depth map untuk diukur."
            } else {
                val distance = engine.sampleTap(result, x, y, depthCalibration)
                tapPoint = x to y
                tapMeasurement = distance
                statusView.text = distance?.let {
                    String.format(
                        Locale.US,
                        "Tap model mentah %.2f m → hasil kalibrasi %.2f m / %.0f cm.",
                        it.modelMeters,
                        it.calibratedMeters,
                        it.calibratedMeters * 100f,
                    )
                } ?: "Tidak ada depth valid di sekitar titik tap."
                redrawLatest()
            }
        }
    }

    private fun updateDisplayWeights() {
        val previewParams = previewHost.layoutParams as LinearLayout.LayoutParams
        val depthParams = depthMapView.layoutParams as LinearLayout.LayoutParams
        previewParams.width = 0
        previewParams.weight = 1f
        depthParams.width = 0
        depthParams.weight = if (sideBySide) 1f else 0f
        previewHost.layoutParams = previewParams
        depthMapView.layoutParams = depthParams
    }

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun bindCamera() {
        if (!hasCameraPermission() || inferenceEngine == null) return
        if (!isBindingCamera.compareAndSet(false, true)) return
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener(
            {
                try {
                    val provider = providerFuture.get()
                    val preview = Preview.Builder()
                        .setTargetRotation(previewView.display?.rotation ?: Surface.ROTATION_0)
                        .build()
                        .also { it.setSurfaceProvider(previewView.surfaceProvider) }
                    val analysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .setTargetRotation(previewView.display?.rotation ?: Surface.ROTATION_0)
                        .build()
                    previewUseCase = preview
                    analysisUseCase = analysis
                    analysis.setAnalyzer(cameraExecutor) { image ->
                        var frame: Bitmap? = null
                        try {
                            val cameraFrame = CameraFrameConverter.toUprightBitmap(image)
                            frame = cameraFrame
                            val result = inferenceEngine?.runFrame(
                                bitmap = cameraFrame,
                                confidenceThreshold = confidenceThreshold,
                                selectedClasses = selectedClasses,
                                calibration = depthCalibration,
                                fps = nextFps(),
                            )
                            if (result != null) {
                                if (markerCalibrationEnabled) {
                                    val focalPixels = markerFocalPixels(image, cameraFrame.width)
                                    if (focalPixels != null) {
                                        scheduleMarkerScan(cameraFrame, result, focalPixels)
                                    } else if (!markerOpticsWarningShown) {
                                        markerOpticsWarningShown = true
                                        runOnUiThread {
                                            calibrationSummaryView.text =
                                                "Kamera tidak memberi metadata lensa; kalibrasi QR otomatis tidak tersedia."
                                        }
                                    }
                                }
                                runOnUiThread { displayResult(result) }
                            }
                        } catch (error: Exception) {
                            runOnUiThread {
                                statusView.text = "Inferensi gagal: ${error.message}"
                            }
                        } finally {
                            frame?.recycle()
                            image.close()
                        }
                    }
                    provider.unbindAll()
                    val viewPort = previewView.viewPort
                    if (viewPort == null) {
                        throw IllegalStateException(
                            "PreviewView belum siap; tidak dapat menyelaraskan crop preview dan analisis.",
                        )
                    }
                    val useCases = UseCaseGroup.Builder()
                        .setViewPort(viewPort)
                        .addUseCase(preview)
                        .addUseCase(analysis)
                        .build()
                    val cameraInfo = provider.getCameraInfo(CameraSelector.DEFAULT_BACK_CAMERA)
                    val camera2Info = Camera2CameraInfo.from(cameraInfo)
                    cameraFocalLengthMm = camera2Info
                        .getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                        ?.firstOrNull()
                    cameraSensorSizeMm = camera2Info
                        .getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, useCases)
                    cameraBound = true
                    statusView.text = "Kamera aktif. Tap gambar untuk mengukur titik."
                    permissionButton.visibility = View.GONE
                } catch (error: Exception) {
                    cameraBound = false
                    statusView.text = "Gagal membuka kamera: ${error.message}"
                } finally {
                    isBindingCamera.set(false)
                }
            },
            ContextCompat.getMainExecutor(this),
        )
    }

    private fun markerFocalPixels(image: androidx.camera.core.ImageProxy, frameWidth: Int): Float? {
        val focalMm = cameraFocalLengthMm ?: return null
        val sensorSize = cameraSensorSizeMm ?: return null
        val crop = image.cropRect
        val isRotated = image.imageInfo.rotationDegrees % 180 != 0
        val physicalFrameWidthMm = if (isRotated) {
            sensorSize.height * crop.height() / image.height
        } else {
            sensorSize.width * crop.width() / image.width
        }
        if (!physicalFrameWidthMm.isFinite() || physicalFrameWidthMm <= 0f) return null
        return focalMm * frameWidth / physicalFrameWidthMm
    }

    private fun scheduleMarkerScan(
        frame: Bitmap,
        result: FrameResult,
        focalPixels: Float,
    ) {
        if (!markerScanPending.compareAndSet(false, true)) return
        val scale = min(1f, MAX_MARKER_SCAN_PIXELS / maxOf(frame.width, frame.height).toFloat())
        val scanWidth = (frame.width * scale).toInt().coerceAtLeast(1)
        val scanHeight = (frame.height * scale).toInt().coerceAtLeast(1)
        val resized = Bitmap.createScaledBitmap(frame, scanWidth, scanHeight, true)
        val scanBitmap = if (resized === frame) {
            frame.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            resized
        }

        try {
            markerScanner.process(InputImage.fromBitmap(scanBitmap, 0))
                .addOnCompleteListener(markerExecutor) { task ->
                    try {
                        if (task.isSuccessful) {
                            val marker = task.result.firstOrNull {
                                it.format == Barcode.FORMAT_QR_CODE &&
                                    it.rawValue == CalibrationMarker.PAYLOAD
                            }
                            if (marker != null) {
                                calibrateFromMarker(
                                    marker,
                                    result,
                                    frame.width,
                                    frame.height,
                                    scanWidth,
                                    scanHeight,
                                    focalPixels,
                                )
                            }
                        } else {
                            android.util.Log.w(TAG, "QR calibration scan failed.", task.exception)
                        }
                    } catch (error: Exception) {
                        android.util.Log.w(TAG, "QR calibration sample rejected.", error)
                    } finally {
                        scanBitmap.recycle()
                        markerScanPending.set(false)
                    }
                }
        } catch (error: Exception) {
            scanBitmap.recycle()
            markerScanPending.set(false)
            android.util.Log.w(TAG, "QR calibration scan could not start.", error)
        }
    }

    private fun calibrateFromMarker(
        marker: Barcode,
        result: FrameResult,
        frameWidth: Int,
        frameHeight: Int,
        scanWidth: Int,
        scanHeight: Int,
        focalPixels: Float,
    ) {
        val corners = marker.cornerPoints?.takeIf { it.size == 4 } ?: return
        val scaleX = frameWidth.toFloat() / scanWidth
        val scaleY = frameHeight.toFloat() / scanHeight
        val points = corners.map { PointF(it.x * scaleX, it.y * scaleY) }
        val sideLengths = points.indices.map { index ->
            val first = points[index]
            val second = points[(index + 1) % points.size]
            hypot((second.x - first.x).toDouble(), (second.y - first.y).toDouble()).toFloat()
        }
        val shortestSide = sideLengths.minOrNull() ?: return
        val longestSide = sideLengths.maxOrNull() ?: return
        if (shortestSide <= 4f || longestSide / shortestSide > MAX_MARKER_SIDE_RATIO) return

        val markerPixels = sideLengths.average().toFloat()
        val actualMeters = focalPixels * CalibrationMarker.SIDE_METERS / markerPixels
        if (!actualMeters.isFinite() || actualMeters !in MIN_MARKER_DISTANCE..MAX_MARKER_DISTANCE) return

        val centerX = points.map { it.x }.average().toFloat()
        val centerY = points.map { it.y }.average().toFloat()
        val halfSampleSize = markerPixels * MARKER_SAMPLE_FRACTION
        val depthScaleX = result.depthWidth / frameWidth.toFloat()
        val depthScaleY = result.depthHeight / frameHeight.toFloat()
        val modelMeters = DepthMath.sampleMedianInBox(
            result.metricDepth,
            result.depthWidth,
            result.depthHeight,
            (centerX - halfSampleSize) * depthScaleX,
            (centerY - halfSampleSize) * depthScaleY,
            (centerX + halfSampleSize) * depthScaleX,
            (centerY + halfSampleSize) * depthScaleY,
        ) ?: return

        val fit = markerAutoCalibrator.observe(actualMeters, modelMeters)
        if (fit == null) {
            runOnUiThread {
                if (markerCalibrationEnabled) {
                    calibrationSummaryView.text = String.format(
                        Locale.US,
                        "Marker %.2f m | depth mentah %.2f m | %s",
                        actualMeters,
                        modelMeters,
                        markerAutoCalibrator.statusText(),
                    )
                }
            }
            return
        }

        val now = android.os.SystemClock.elapsedRealtime()
        if (now - lastAutoCalibrationSaveMs < AUTO_CALIBRATION_SAVE_INTERVAL_MS) return
        lastAutoCalibrationSaveMs = now
        depthCalibration = fit.calibration
        inferenceEngine?.resetDistanceSmoothing()
        calibrationPreferences.edit()
            .putFloat(CALIBRATION_A, fit.calibration.a)
            .putFloat(CALIBRATION_B, fit.calibration.b)
            .putString(CALIBRATION_PAIRS, LinearDepthCalibration.serializePairs(fit.pairs))
            .apply()
        val pairsText = LinearDepthCalibration.serializePairs(fit.pairs)
        runOnUiThread {
            if (markerCalibrationEnabled) {
                calibrationPairsInput.setText(pairsText)
                calibrationSummaryView.text = String.format(
                    Locale.US,
                    "Auto QR: y = %.4f × model + %.4f m · R² %.2f · galat %.2f m · %d jarak",
                    fit.calibration.a,
                    fit.calibration.b,
                    fit.rSquared,
                    fit.rmseMeters,
                    fit.pairs.size,
                )
                statusView.text = "Kalibrasi otomatis diperbarui dari marker QR 10 cm."
            }
        }
    }

    private fun nextFps(): Float {
        val now = System.nanoTime()
        if (lastFpsWindowNs == 0L) lastFpsWindowNs = now
        framesInFpsWindow++
        val elapsed = now - lastFpsWindowNs
        if (elapsed >= 1_000_000_000L) {
            displayedFps = framesInFpsWindow * 1_000_000_000f / elapsed
            framesInFpsWindow = 0
            lastFpsWindowNs = now
        }
        return displayedFps
    }

    private fun displayResult(result: FrameResult) {
        latestResult = result
        latestDepthBitmap?.recycle()
        latestDepthBitmap = Bitmap.createBitmap(
            result.depthWidth,
            result.depthHeight,
            Bitmap.Config.ARGB_8888,
        ).apply {
            setPixels(
                result.depthColor,
                0,
                result.depthWidth,
                0,
                0,
                result.depthWidth,
                result.depthHeight,
            )
        }
        redrawLatest()
        statusView.text = String.format(
            Locale.US,
            "%.1f FPS ? %.1f ms ? %d box ? score %.2f ? %d > conf ? %d output model ? %s ? %s",
            result.fps,
            result.inferenceLatencyMs,
            result.detections.size,
            result.detectorMaxScore,
            result.detectorCandidatesAboveThreshold,
            result.detectorValidBeforeNms,
            if (result.detectorCoordinatesNormalized) "norm" else "px",
            result.runtimeLabel,
        )
    }

    private fun redrawLatest() {
        val result = latestResult ?: return
        overlayView.update(
            result,
            latestDepthBitmap,
            showDepthOverlay && !sideBySide,
            tapPoint,
            tapMeasurement?.modelMeters,
            tapMeasurement?.calibratedMeters,
        )
        result.detections
            .sortedByDescending { it.confidence }
            .take(3)
            .forEach { detection ->
                val screenBox = overlayView.mapFrameBoxToView(detection)
                android.util.Log.i(
                    "YoloBoxes",
                    "${detection.className} score=${detection.confidence} " +
                        "raw_xywh=[${detection.rawCenterX},${detection.rawCenterY}," +
                        "${detection.rawWidth},${detection.rawHeight}] " +
                        "frame_xyxy=[${detection.left},${detection.top}," +
                        "${detection.right},${detection.bottom}] " +
                        "overlay_px=${screenBox?.let {
                            "[${it.left},${it.top},${it.right},${it.bottom}]"
                        }}",
                )
            }
        depthMapView.update(
            if (sideBySide) latestResult else null,
            if (sideBySide) latestDepthBitmap else null,
        )
    }

    private fun parseClasses(value: String): Set<String> =
        value.split(",")
            .map { it.trim().lowercase(Locale.ROOT) }
            .filter { it.isNotEmpty() }
            .map { if (it == "laptop") "computer" else it }
            .toSet()

    private fun classFilterStatus(classes: Set<String>): String {
        if (classes.isEmpty()) return "Filter kelas dikosongkan; menampilkan semua kelas."
        val knownClasses = config.classNames.map { it.lowercase(Locale.ROOT) }.toSet()
        val unknown = classes - knownClasses
        return if (unknown.isEmpty()) {
            "Filter aktif: ${classes.joinToString()}"
        } else {
            "Kelas tidak dikenal pada model: ${unknown.joinToString()}"
        }
    }

    private fun fitAndSaveCalibration() {
        try {
            val pairs = LinearDepthCalibration.parsePairs(
                calibrationPairsInput.text.toString(),
            )
            val fitted = LinearDepthCalibration.fit(pairs)
            val saved = calibrationPreferences.edit()
                .putFloat(CALIBRATION_A, fitted.a)
                .putFloat(CALIBRATION_B, fitted.b)
                .putString(CALIBRATION_PAIRS, LinearDepthCalibration.serializePairs(pairs))
                .commit()
            if (!saved) throw IOException("SharedPreferences gagal menyimpan kalibrasi.")

            depthCalibration = fitted
            inferenceEngine?.resetDistanceSmoothing()
            tapMeasurement = null
            calibrationSummaryView.text = calibrationDescription(pairs.size)
            statusView.text = String.format(
                Locale.US,
                "Kalibrasi tersimpan: jarak = %.4f × model + %.4f; %d pasangan. " +
                    "Khusus kamera dan pipeline validasi ini.",
                fitted.a,
                fitted.b,
                pairs.size,
            )
        } catch (error: IllegalArgumentException) {
            statusView.text = "Kalibrasi gagal: ${error.message}"
        } catch (error: IOException) {
            android.util.Log.e(TAG, "Gagal menyimpan kalibrasi.", error)
            statusView.text = "Kalibrasi gagal disimpan: ${error.message}"
        }
    }

    private fun resetCalibrationToRaw() {
        val saved = calibrationPreferences.edit()
            .putFloat(CALIBRATION_A, 1f)
            .putFloat(CALIBRATION_B, 0f)
            .putString(CALIBRATION_PAIRS, "")
            .commit()
        if (!saved) {
            statusView.text = "Reset kalibrasi gagal disimpan."
            return
        }
        depthCalibration = LinearDepthCalibration.IDENTITY
        markerAutoCalibrator.reset()
        calibrationPairsInput.text.clear()
        inferenceEngine?.resetDistanceSmoothing()
        tapMeasurement = null
        calibrationSummaryView.text = calibrationDescription(0)
        redrawLatest()
        statusView.text = "Kalibrasi direset; jarak menampilkan nilai model mentah."
    }

    private fun exportCalibrationMarker(uri: Uri) {
        try {
            val output = contentResolver.openOutputStream(uri)
                ?: throw IOException("File marker tujuan tidak dapat dibuka.")
            output.use { CalibrationMarker.writePdf(it) }
            statusView.text = "PDF marker QR disimpan. Cetak ukuran sebenarnya/100%, tanpa fit-to-page."
        } catch (error: Exception) {
            android.util.Log.e(TAG, "Gagal menulis marker kalibrasi.", error)
            statusView.text = "Ekspor marker gagal: ${error.message}"
        }
    }

    private fun calibrationDescription(pairCount: Int? = null): String {
        val count = pairCount ?: LinearDepthCalibration.parsePairs(
            calibrationPreferences.getString(CALIBRATION_PAIRS, "").orEmpty(),
        ).size
        if (count == 0) return "Jarak model mentah (belum dikalibrasi)"
        return String.format(
            Locale.US,
            "Jarak = %.4f × model + %.4f m · %d pasangan tersimpan",
            depthCalibration.a,
            depthCalibration.b,
            count,
        )
    }

    private fun exportCalibrationCsv(uri: Uri) {
        try {
            val pairs = LinearDepthCalibration.parsePairs(
                calibrationPairsInput.text.toString(),
            )
            if (pairs.isEmpty()) {
                throw IllegalArgumentException("Masukkan sedikitnya satu pasangan untuk diekspor.")
            }
            val output = contentResolver.openOutputStream(uri)
                ?: throw IOException("File tujuan tidak dapat dibuka.")
            OutputStreamWriter(output, Charsets.UTF_8).use { writer ->
                writer.appendLine(
                    "jarak_asli_m,jarak_model_m,jarak_terkalibrasi_m",
                )
                for (pair in pairs) {
                    writer.appendLine(
                        String.format(
                            Locale.US,
                            "%.6f,%.6f,%.6f",
                            pair.actualMeters,
                            pair.modelMeters,
                            depthCalibration.apply(pair.modelMeters),
                        ),
                    )
                }
            }
            statusView.text = "CSV berhasil diekspor (${pairs.size} pasangan)."
            android.widget.Toast.makeText(
                this,
                "CSV validasi berhasil disimpan.",
                android.widget.Toast.LENGTH_LONG,
            ).show()
        } catch (error: IllegalArgumentException) {
            statusView.text = "Ekspor CSV gagal: ${error.message}"
        } catch (error: IOException) {
            android.util.Log.e(TAG, "Gagal menulis CSV kalibrasi.", error)
            statusView.text = "Ekspor CSV gagal: ${error.message}"
        } catch (error: SecurityException) {
            android.util.Log.e(TAG, "Izin menulis CSV ditolak.", error)
            statusView.text = "Ekspor CSV gagal: akses file ditolak."
        }
    }

    private fun text(value: String, sizeSp: Float): TextView =
        TextView(this).apply {
            text = value
            textSize = sizeSp
            setTextColor(0xffffffff.toInt())
        }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density).toInt()

    private fun Float.format(decimals: Int): String =
        String.format(Locale.US, "%.${decimals}f", this)

    private fun showFatalError(message: String) {
        android.util.Log.e(TAG, message)
        android.widget.Toast.makeText(this, message, android.widget.Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            inferenceEngine?.close()
        } catch (error: Exception) {
            android.util.Log.w(TAG, "Gagal menutup sesi ONNX Runtime dengan bersih.", error)
        }
        if (::cameraExecutor.isInitialized) cameraExecutor.shutdown()
        if (::markerScanner.isInitialized) markerScanner.close()
        if (::markerExecutor.isInitialized) markerExecutor.shutdown()
        latestDepthBitmap?.recycle()
    }

    companion object {
        private const val TAG = "Yolo26Depth"
        private const val DEFAULT_CONFIDENCE = 0.05f
        private const val MAX_CONFIDENCE = 0.80f
        private const val CALIBRATION_PREFERENCES = "linear_depth_calibration"
        private const val CALIBRATION_A = "a"
        private const val CALIBRATION_B = "b"
        private const val CALIBRATION_PAIRS = "pairs"
        private const val CALIBRATION_PROFILE_VERSION = "profile_version"
        private const val MAX_MARKER_SCAN_PIXELS = 960f
        private const val MAX_MARKER_SIDE_RATIO = 1.35f
        private const val MARKER_SAMPLE_FRACTION = 0.20f
        private const val MIN_MARKER_DISTANCE = 0.25f
        private const val MAX_MARKER_DISTANCE = 8f
        private const val AUTO_CALIBRATION_SAVE_INTERVAL_MS = 1500L
    }
}
