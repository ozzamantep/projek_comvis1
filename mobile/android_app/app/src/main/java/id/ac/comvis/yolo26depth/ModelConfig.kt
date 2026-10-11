package id.ac.comvis.yolo26depth

import android.content.Context
import org.json.JSONObject

data class ModelConfig(
    val depthModel: String,
    val depthInputWidth: Int,
    val depthInputHeight: Int,
    val depthInputLayout: String,
    val depthInputDtype: String,
    val depthOutputLayout: String,
    val depthOutputChannels: Int,
    val depthScaleMeters: Float,
    val calibrationFactor: Float,
    val depthPatchSize: Int,
    val emaAlpha: Float,
    val detectorModel: String,
    val detectorInputSize: Int,
    val detectorInputLayout: String,
    val detectorOutputChannels: Int,
    val detectorCandidates: Int,
    val detectorClassCount: Int,
    val detectorConfidence: Float,
    val detectorIouThreshold: Float,
    val detectorDecodeVerified: Boolean,
) {
    init {
        require(depthInputWidth > 0 && depthInputHeight > 0)
        require(depthInputLayout == "NHWC" && depthOutputLayout == "NHWC")
        require(depthInputDtype == "FLOAT32" && depthOutputChannels == 1)
        require(depthScaleMeters > 0f && calibrationFactor > 0f)
        require(depthPatchSize > 0 && depthPatchSize % 2 == 1)
        require(emaAlpha > 0f && emaAlpha <= 1f)
        require(detectorInputLayout == "NCHW" && detectorInputSize > 0)
        require(detectorOutputChannels == detectorClassCount + 4)
        require(detectorCandidates > 0 && detectorIouThreshold in 0f..1f)
    }

    companion object {
        fun load(context: Context): ModelConfig {
            val json = context.assets.open("model_config.json")
                .bufferedReader()
                .use { JSONObject(it.readText()) }
            return ModelConfig(
                depthModel = json.getString("depth_model"),
                depthInputWidth = json.getInt("depth_input_width"),
                depthInputHeight = json.getInt("depth_input_height"),
                depthInputLayout = json.getString("depth_input_layout"),
                depthInputDtype = json.getString("depth_input_dtype"),
                depthOutputLayout = json.getString("depth_output_layout"),
                depthOutputChannels = json.getInt("depth_output_channels"),
                depthScaleMeters = json.getDouble("depth_scale_m").toFloat(),
                calibrationFactor = json.getDouble("calibration_factor").toFloat(),
                depthPatchSize = json.getInt("depth_patch_size"),
                emaAlpha = json.getDouble("ema_alpha").toFloat(),
                detectorModel = json.getString("detector_model"),
                detectorInputSize = json.getInt("detector_input_size"),
                detectorInputLayout = json.getString("detector_input_layout"),
                detectorOutputChannels = json.getInt("detector_output_channels"),
                detectorCandidates = json.getInt("detector_candidates"),
                detectorClassCount = json.getInt("detector_class_count"),
                detectorConfidence = json.getDouble("detector_confidence").toFloat(),
                detectorIouThreshold = json.getDouble("detector_iou_threshold").toFloat(),
                detectorDecodeVerified = json.getBoolean("detector_decode_verified"),
            )
        }
    }
}
