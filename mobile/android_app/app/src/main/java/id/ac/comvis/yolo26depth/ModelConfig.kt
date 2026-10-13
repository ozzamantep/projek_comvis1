package id.ac.comvis.yolo26depth

import android.content.Context
import org.json.JSONObject

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
