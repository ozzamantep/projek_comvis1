package id.ac.comvis.yolo26depth

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.Closeable

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
