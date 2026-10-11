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
        val boxes = ArrayList<ModelBox>()
        var maxScore = 0f
        var candidatesAboveThreshold = 0
        for (index in 0 until candidates) {
            var classId = -1
            var score = 0f
            for (classIndex in 0 until config.detectorClassCount) {
                val candidateScore = output[(classIndex + 4) * candidates + index]
                if (candidateScore > score) {
                    classId = classIndex
                    score = candidateScore
                }
            }
            maxScore = maxOf(maxScore, score)
            if (classId < 0 || score < confidenceThreshold) continue
            candidatesAboveThreshold++
            if (selectedClasses.isNotEmpty()) {
                val label = CocoLabels.names.getOrNull(classId)?.lowercase() ?: continue
                if (label !in selectedClasses) continue
            }

            val centerX = output[index]
            val centerY = output[candidates + index]
            val width = output[2 * candidates + index]
            val height = output[3 * candidates + index]
            val left = ((centerX - width / 2f - transform.padX) / transform.scale)
                .coerceIn(0f, transform.originalWidth.toFloat())
            val top = ((centerY - height / 2f - transform.padY) / transform.scale)
                .coerceIn(0f, transform.originalHeight.toFloat())
            val right = ((centerX + width / 2f - transform.padX) / transform.scale)
                .coerceIn(0f, transform.originalWidth.toFloat())
            val bottom = ((centerY + height / 2f - transform.padY) / transform.scale)
                .coerceIn(0f, transform.originalHeight.toFloat())
            if (right > left && bottom > top) {
                boxes += ModelBox(classId, score, left, top, right, bottom)
            }
        }

        val sorted = boxes.sortedByDescending { it.confidence }.take(MAX_PRE_NMS)
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
        return DetectionDecodeResult(
            boxes = kept,
            maxScore = maxScore,
            candidatesAboveThreshold = candidatesAboveThreshold,
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
}
