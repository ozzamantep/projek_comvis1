package id.ac.comvis.yolo26depth

object CocoLabels {
    val names = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    )
}

object YoloPostprocessor {
    private const val MAX_PRE_NMS = 300
    private const val MAX_DETECTIONS = 100
    private const val DEBUG_LOG_INTERVAL_MS = 1_000L
    private var lastDebugLogMs = 0L

    private data class DebugCandidate(
        val classId: Int,
        val score: Float,
        val rawXywh: FloatArray,
        val frameXyxy: FloatArray,
    )

    fun decode(
        output: FloatArray,
        config: ModelConfig,
        transform: LetterboxTransform,
        confidenceThreshold: Float,
        selectedClasses: Set<String>,
    ): DetectionDecodeResult {
        val candidates = config.detectorCandidates
        val channels = config.detectorOutputChannels
        require(output.size == channels * candidates) {
            "Detector output length ${output.size}; expected ${channels * candidates}."
        }
        val coordinatesMaxAbs = (0 until 4).maxOf { channel ->
            (0 until candidates).maxOf { index ->
                kotlin.math.abs(output[channel * candidates + index])
            }
        }
        val coordinatesNormalized = coordinatesMaxAbs <= 2f && coordinatesMaxAbs > 0f
        val coordinateScale = if (coordinatesNormalized) config.detectorInputSize.toFloat() else 1f
        val boxesBeforeNms = ArrayList<ModelBox>()
        val topCandidates = ArrayList<DebugCandidate>(3)
        var maxScore = 0f
        var candidatesAboveThreshold = 0
        var scoreMax = 0f

        for (index in 0 until candidates) {
            var classId = -1
            var score = 0f
            for (classIndex in 0 until config.detectorClassCount) {
                // Exported class scores are consumed directly; applying sigmoid again would distort them.
                val candidateScore = output[(classIndex + 4) * candidates + index]
                if (candidateScore > score) {
                    classId = classIndex
                    score = candidateScore
                }
            }
            if (score.isFinite()) {
                scoreMax = maxOf(scoreMax, score)
                maxScore = maxOf(maxScore, score)
            }
            if (classId >= 0 && score.isFinite()) {
                val rawXywh = floatArrayOf(
                    output[index],
                    output[candidates + index],
                    output[2 * candidates + index],
                    output[3 * candidates + index],
                )
                val candidate = toDebugCandidate(
                    classId,
                    score,
                    rawXywh,
                    coordinateScale,
                    transform,
                )
                if (topCandidates.size < 3) {
                    topCandidates += candidate
                    topCandidates.sortByDescending { it.score }
                } else if (score > topCandidates.last().score) {
                    topCandidates[topCandidates.lastIndex] = candidate
                    topCandidates.sortByDescending { it.score }
                }
            }
            if (classId < 0 || score < confidenceThreshold) continue
            if (!score.isFinite()) continue
            candidatesAboveThreshold++
            if (selectedClasses.isNotEmpty()) {
                val label = CocoLabels.names.getOrNull(classId)?.lowercase() ?: continue
                if (label !in selectedClasses) continue
            }

            val centerX = output[index] * coordinateScale
            val centerY = output[candidates + index] * coordinateScale
            val width = output[2 * candidates + index] * coordinateScale
            val height = output[3 * candidates + index] * coordinateScale
            if (!centerX.isFinite() || !centerY.isFinite() ||
                !width.isFinite() || !height.isFinite() || width <= 0f || height <= 0f
            ) {
                continue
            }
            val left = ((centerX - width / 2f - transform.padX) / transform.scale)
                .coerceIn(0f, transform.originalWidth.toFloat())
            val top = ((centerY - height / 2f - transform.padY) / transform.scale)
                .coerceIn(0f, transform.originalHeight.toFloat())
            val right = ((centerX + width / 2f - transform.padX) / transform.scale)
                .coerceIn(0f, transform.originalWidth.toFloat())
            val bottom = ((centerY + height / 2f - transform.padY) / transform.scale)
                .coerceIn(0f, transform.originalHeight.toFloat())
            if (right > left && bottom > top) {
                    boxesBeforeNms += ModelBox(
                        classId = classId,
                        confidence = score,
                        rawCenterX = output[index],
                        rawCenterY = output[candidates + index],
                        rawWidth = output[2 * candidates + index],
                        rawHeight = output[3 * candidates + index],
                        left = left,
                        top = top,
                        right = right,
                        bottom = bottom,
                    )
                }
        }

        val sorted = boxesBeforeNms.sortedByDescending { it.confidence }.take(MAX_PRE_NMS)
        val kept = ArrayList<ModelBox>()
        for (candidate in sorted) {
            if (kept.none {
                    it.classId == candidate.classId &&
                        intersectionOverUnion(it, candidate) > config.detectorIouThreshold
                }
            ) {
                kept += candidate
                if (kept.size >= MAX_DETECTIONS) break
            }
        }
        logDecodeSummary(
            channels = channels,
            candidates = candidates,
            scoreMax = scoreMax,
            coordinatesMaxAbs = coordinatesMaxAbs,
            coordinatesNormalized = coordinatesNormalized,
            confidenceThreshold = confidenceThreshold,
            selectedClasses = selectedClasses,
            candidatesAboveThreshold = candidatesAboveThreshold,
            boxesBeforeNms = boxesBeforeNms.size,
            afterNms = kept.size,
            topCandidates = topCandidates,
        )
        return DetectionDecodeResult(
            boxes = kept,
            maxScore = maxScore,
            candidatesAboveThreshold = candidatesAboveThreshold,
            validBoxesBeforeNms = boxesBeforeNms.size,
            coordinatesNormalized = coordinatesNormalized,
        )
    }

    private fun toDebugCandidate(
        classId: Int,
        score: Float,
        rawXywh: FloatArray,
        coordinateScale: Float,
        transform: LetterboxTransform,
    ): DebugCandidate {
        val cx = rawXywh[0] * coordinateScale
        val cy = rawXywh[1] * coordinateScale
        val width = rawXywh[2] * coordinateScale
        val height = rawXywh[3] * coordinateScale
        val frameXyxy = floatArrayOf(
            (cx - width / 2f - transform.padX) / transform.scale,
            (cy - height / 2f - transform.padY) / transform.scale,
            (cx + width / 2f - transform.padX) / transform.scale,
            (cy + height / 2f - transform.padY) / transform.scale,
        )
        return DebugCandidate(classId, score, rawXywh, frameXyxy)
    }

    private fun logDecodeSummary(
        channels: Int,
        candidates: Int,
        scoreMax: Float,
        coordinatesMaxAbs: Float,
        coordinatesNormalized: Boolean,
        confidenceThreshold: Float,
        selectedClasses: Set<String>,
        candidatesAboveThreshold: Int,
        boxesBeforeNms: Int,
        afterNms: Int,
        topCandidates: List<DebugCandidate>,
    ) {
        val now = android.os.SystemClock.elapsedRealtime()
        if (now - lastDebugLogMs < DEBUG_LOG_INTERVAL_MS) return
        lastDebugLogMs = now
        val top = topCandidates.joinToString(separator = " | ") { candidate ->
            val raw = candidate.rawXywh.joinToString(",") { "%.2f".format(java.util.Locale.US, it) }
            val frame = candidate.frameXyxy.joinToString(",") { "%.1f".format(java.util.Locale.US, it) }
            "${CocoLabels.names[candidate.classId]} score=${"%.3f".format(java.util.Locale.US, candidate.score)} " +
                "raw_xywh=[$raw] frame_xyxy=[$frame]"
        }
        android.util.Log.i(
            TAG,
            "shape=[1,$channels,$candidates] scoreMax=$scoreMax coordsMaxAbs=$coordinatesMaxAbs " +
                "coords=${if (coordinatesNormalized) "normalized" else "pixels"} " +
                "conf=$confidenceThreshold classFilter=${selectedClasses.ifEmpty { setOf("ALL") }} " +
                "aboveConf=$candidatesAboveThreshold validBeforeNms=$boxesBeforeNms afterNms=$afterNms " +
                "top3={$top}",
        )
    }

    private fun intersectionOverUnion(a: ModelBox, b: ModelBox): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        val intersection = (right - left).coerceAtLeast(0f) *
            (bottom - top).coerceAtLeast(0f)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        return intersection / (areaA + areaB - intersection).coerceAtLeast(1e-6f)
    }

    private const val TAG = "YoloDecode"
}
