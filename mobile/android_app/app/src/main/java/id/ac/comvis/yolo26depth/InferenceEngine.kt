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

    val runtimeLabel: String
        get() = "Depth ${depthModel.label} · Detektor ${detectorModel.label}"

    init {
        val depthInput = depthModel.interpreter.getInputTensor(0).shape()
        val depthOutput = depthModel.interpreter.getOutputTensor(0).shape()
        require(depthInput.contentEquals(
            intArrayOf(1, config.depthInputHeight, config.depthInputWidth, 3),
        )) {
            "Depth input $depthInput, expected [1,${config.depthInputHeight}," +
                "${config.depthInputWidth},3] NHWC."
        }
        require(depthOutput.contentEquals(
            intArrayOf(1, config.depthInputHeight, config.depthInputWidth, 1),
        )) {
            "Depth output $depthOutput, expected [1,${config.depthInputHeight}," +
                "${config.depthInputWidth},1] NHWC."
        }

        val detectorInput = detectorModel.interpreter.getInputTensor(0).shape()
        val detectorOutput = detectorModel.interpreter.getOutputTensor(0).shape()
        require(detectorInput.contentEquals(
            intArrayOf(1, 3, config.detectorInputSize, config.detectorInputSize),
        )) {
            "Detector input $detectorInput does not match configured NCHW input."
        }
        require(detectorOutput.contentEquals(
            intArrayOf(
                1,
                config.detectorOutputChannels,
                config.detectorCandidates,
            ),
        )) {
            "Detector output $detectorOutput does not match configured BCN output."
        }
        require(depthModel.interpreter.getInputTensor(0).dataType().name == "FLOAT32")
        require(detectorModel.interpreter.getInputTensor(0).dataType().name == "FLOAT32")
    }

    fun runFrame(
        bitmap: Bitmap,
        confidenceThreshold: Float,
        selectedClasses: Set<String>,
        calibrationFactor: Float,
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

        val depthInput = ImagePreprocessor.depthInput(bitmap, config)
        val depthStart = SystemClock.elapsedRealtimeNanos()
        val depthRaw = runModel(
            depthModel,
            depthInput,
            config.depthInputWidth * config.depthInputHeight,
        )
        val depthLatencyMs =
            (SystemClock.elapsedRealtimeNanos() - depthStart) / 1_000_000f
        val metricDepth = ema.update(
            FloatArray(depthRaw.size) { index ->
                depthRaw[index] * config.depthScaleMeters
            },
        )

        val detections = decodeResult.boxes.map { box ->
            val depthX = boxCenterX(box) * config.depthInputWidth / bitmap.width
            val depthY = boxCenterY(box) * config.depthInputHeight / bitmap.height
            val sampled = DepthMath.sampleMedian(
                metricDepth,
                config.depthInputWidth,
                config.depthInputHeight,
                depthX,
                depthY,
                config.depthPatchSize,
            )
            Detection(
                classId = box.classId,
                className = CocoLabels.names[box.classId],
                confidence = box.confidence,
                left = box.left,
                top = box.top,
                right = box.right,
                bottom = box.bottom,
                distanceMeters = sampled?.times(calibrationFactor),
            )
        }

        return FrameResult(
            width = bitmap.width,
            height = bitmap.height,
            detections = detections,
            metricDepth = metricDepth,
            depthWidth = config.depthInputWidth,
            depthHeight = config.depthInputHeight,
            depthColor = InfernoPalette.colorizeForDisplay(metricDepth),
            inferenceLatencyMs = detectorLatencyMs + depthLatencyMs,
            fps = fps,
            runtimeLabel = runtimeLabel,
            detectorMaxScore = decodeResult.maxScore,
            detectorCandidatesAboveThreshold = decodeResult.candidatesAboveThreshold,
        )
    }

    fun sampleTap(
        result: FrameResult,
        frameX: Float,
        frameY: Float,
        calibrationFactor: Float,
    ): Float? {
        val x = frameX * result.depthWidth / result.width
        val y = frameY * result.depthHeight / result.height
        return DepthMath.sampleMedian(
            result.metricDepth,
            result.depthWidth,
            result.depthHeight,
            x,
            y,
            config.depthPatchSize,
        )?.times(calibrationFactor)
    }

    private fun boxCenterX(box: ModelBox): Float = (box.left + box.right) / 2f

    private fun boxCenterY(box: ModelBox): Float = (box.top + box.bottom) / 2f

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
