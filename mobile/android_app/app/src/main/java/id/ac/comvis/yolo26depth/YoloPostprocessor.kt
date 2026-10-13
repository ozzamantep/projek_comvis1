package id.ac.comvis.yolo26depth

import kotlin.math.roundToInt

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
