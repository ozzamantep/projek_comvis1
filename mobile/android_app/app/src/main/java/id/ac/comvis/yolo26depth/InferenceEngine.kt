package id.ac.comvis.yolo26depth

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

class InferenceEngine(
    context: Context,
    private val config: ModelConfig,
) : Closeable {
    private data class RuntimeModel(
        val interpreter: Interpreter,
        val delegate: GpuDelegate?,
        val label: String,
    )

    private val appContext = context.applicationContext
    private val depthModel = createInterpreter(config.depthModel)
    private val detectorModel = createInterpreter(config.detectorModel)
    private val ema = DepthEma(config.emaAlpha)
    private val depthHistory = DepthFrameHistory(maxFrames = 7)
    private val distanceSmoother = DetectionDistanceSmoother(maxFrames = 7)

    val runtimeLabel: String
        get() = "Depth ${depthModel.label} · Detektor ${detectorModel.label}"

    init {
        val depthInput = depthModel.interpreter.getInputTensor(0).shape()
        val depthOutput = depthModel.interpreter.getOutputTensor(0).shape()
        val expectedDepthInput = if (config.depthInputLayout == "NCHW") {
            intArrayOf(1, 3, config.depthInputHeight, config.depthInputWidth)
        } else {
            intArrayOf(1, config.depthInputHeight, config.depthInputWidth, 3)
        }
        require(depthInput.contentEquals(expectedDepthInput)) {
            "Depth input ${depthInput.contentToString()}, expected " +
                "${expectedDepthInput.contentToString()} ${config.depthInputLayout}."
        }
        val expectedDepthOutput = if (config.depthOutputLayout == "NCHW") {
            intArrayOf(1, config.depthOutputChannels, config.depthInputHeight, config.depthInputWidth)
        } else {
            intArrayOf(1, config.depthInputHeight, config.depthInputWidth, config.depthOutputChannels)
        }
        require(depthOutput.contentEquals(expectedDepthOutput)) {
            "Depth output ${depthOutput.contentToString()}, expected " +
                "${expectedDepthOutput.contentToString()} ${config.depthOutputLayout}."
        }

        val detectorInput = detectorModel.interpreter.getInputTensor(0).shape()
        val detectorOutput = detectorModel.interpreter.getOutputTensor(0).shape()
        require(detectorInput.contentEquals(
            intArrayOf(1, 3, config.detectorInputSize, config.detectorInputSize),
        )) {
            "Detector input ${detectorInput.contentToString()} does not match configured NCHW input."
        }
        require(detectorOutput.contentEquals(
            intArrayOf(
                1,
                config.detectorOutputChannels,
                config.detectorCandidates,
            ),
        )) {
            "Detector output ${detectorOutput.contentToString()} does not match configured BCN output."
        }
        require(depthModel.interpreter.getInputTensor(0).dataType().name == "FLOAT32")
        require(detectorModel.interpreter.getInputTensor(0).dataType().name == "FLOAT32")
    }

    fun runFrame(
        bitmap: Bitmap,
        confidenceThreshold: Float,
        selectedClasses: Set<String>,
        calibration: LinearDepthCalibration,
        fps: Float,
    ): FrameResult {
        val (detectorInput, letterbox) = ImagePreprocessor.detectorInput(
            bitmap,
            config.detectorInputSize,
        )
        val detectorStart = SystemClock.elapsedRealtimeNanos()
        val detectorOutput = runModel(
            detectorModel,
            detectorInput,
            config.detectorOutputChannels * config.detectorCandidates,
        )
        val detectorLatencyMs =
            (SystemClock.elapsedRealtimeNanos() - detectorStart) / 1_000_000f

        val decodeResult = YoloPostprocessor.decode(
            output = detectorOutput,
            config = config,
            transform = letterbox,
            confidenceThreshold = confidenceThreshold,
            selectedClasses = selectedClasses,
        )

        val (depthInput, depthTransform) = ImagePreprocessor.depthInput(bitmap, config)
        val depthStart = SystemClock.elapsedRealtimeNanos()
        val depthRaw = runModel(
            depthModel,
            depthInput,
            config.depthInputWidth * config.depthInputHeight,
        )
        val depthLatencyMs =
            (SystemClock.elapsedRealtimeNanos() - depthStart) / 1_000_000f
        val metricDepthSquare = FloatArray(depthRaw.size) { index ->
            depthRaw[index] * config.depthScaleMeters
        }
        val croppedDepth = DepthMath.removeLetterbox(
            metricDepthSquare,
            config.depthInputWidth,
            config.depthInputHeight,
            depthTransform,
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
                className = CocoLabels.names[box.classId],
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
            inferenceLatencyMs = detectorLatencyMs + depthLatencyMs,
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

    private fun runModel(
        runtime: RuntimeModel,
        input: ByteBuffer,
        outputFloatCount: Int,
    ): FloatArray {
        val output = ByteBuffer.allocateDirect(outputFloatCount * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        input.rewind()
        runtime.interpreter.run(input, output)
        output.rewind()
        val floatBuffer = output.asFloatBuffer()
        return FloatArray(outputFloatCount).also(floatBuffer::get)
    }

    private fun createInterpreter(assetName: String): RuntimeModel {
        val modelBuffer = appContext.assets.open(assetName).use { input ->
            val bytes = input.readBytes()
            ByteBuffer.allocateDirect(bytes.size)
                .order(ByteOrder.nativeOrder())
                .apply {
                    put(bytes)
                    rewind()
                }
        }

        var gpuDelegate: GpuDelegate? = null
        var gpuInterpreter: Interpreter? = null
        try {
            gpuDelegate = GpuDelegate()
            val gpuOptions = Interpreter.Options()
                .addDelegate(gpuDelegate)
                .setNumThreads(4)
            val initializedInterpreter = Interpreter(modelBuffer, gpuOptions)
            gpuInterpreter = initializedInterpreter
            initializedInterpreter.allocateTensors()
            return RuntimeModel(initializedInterpreter, gpuDelegate, "GPU")
        } catch (gpuError: Exception) {
            gpuInterpreter?.close()
            gpuDelegate?.close()
            logGpuFallback(assetName, gpuError)
        } catch (gpuError: LinkageError) {
            gpuInterpreter?.close()
            gpuDelegate?.close()
            logGpuFallback(assetName, gpuError)
        }

        val cpuOptions = Interpreter.Options()
            .setNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 4))
            .setUseXNNPACK(true)
        val cpuInterpreter = Interpreter(modelBuffer, cpuOptions)
        cpuInterpreter.allocateTensors()
        return RuntimeModel(cpuInterpreter, null, "CPU/XNNPACK")
    }

    private fun logGpuFallback(assetName: String, error: Throwable) {
        android.util.Log.w(
            TAG,
            "GPU delegate unavailable for $assetName; retrying on CPU/XNNPACK.",
            error,
        )
    }

    override fun close() {
        depthModel.interpreter.close()
        detectorModel.interpreter.close()
        depthModel.delegate?.close()
        detectorModel.delegate?.close()
    }

    companion object {
        private const val TAG = "Yolo26Depth"
    }
}
