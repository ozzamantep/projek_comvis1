package id.ac.comvis.yolo26depth

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
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
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : androidx.activity.ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: DetectionOverlayView
    private lateinit var depthMapView: DepthMapView
    private lateinit var statusView: TextView
    private lateinit var warningView: TextView
    private lateinit var permissionButton: Button
    private lateinit var frameRow: LinearLayout
    private lateinit var previewHost: android.widget.FrameLayout
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var config: ModelConfig
    private var inferenceEngine: InferenceEngine? = null
    private var latestResult: FrameResult? = null
    private var latestDepthBitmap: Bitmap? = null
    private var tapPoint: Pair<Float, Float>? = null
    private var tapDistanceMeters: Float? = null
    private var showDepthOverlay = false
    private var sideBySide = false
    @Volatile private var confidenceThreshold = 0.5f
    @Volatile private var selectedClasses: Set<String> = emptySet()
    @Volatile private var calibrationFactor = 1f
    private var lastFpsWindowNs = 0L
    private var framesInFpsWindow = 0
    private var displayedFps = 0f
    private val isBindingCamera = AtomicBoolean(false)
    private var permissionPermanentlyDenied = false
    private var startupError: String? = null
    private var cameraBound = false

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
        try {
            config = ModelConfig.load(this)
            calibrationFactor = config.calibrationFactor
            confidenceThreshold = config.detectorConfidence
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

    private fun createUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xff101010.toInt())
        }

        warningView = text(
            "Estimasi, bukan alat ukur presisi. Detector decode/NMS BELUM diverifikasi " +
                "terhadap Ultralytics PC pada gambar nyata.",
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
        depthMapView.visibility = View.GONE
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
            progress = (confidenceThreshold * 100).toInt()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    confidenceThreshold = progress / 100f
                    confidenceLabel.text = "Confidence: ${confidenceThreshold.format(2)}"
                }

                override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
            })
        }
        controls.addView(confidenceLabel)
        controls.addView(confidenceSeekBar)

        val filterRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        filterRow.addView(text("Kelas (kosong = semua):", 12f))
        val classFilter = EditText(this).apply {
            setSingleLine(true)
            hint = "person, cell phone"
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
        filterRow.addView(
            classFilter,
            LinearLayout.LayoutParams(0, dp(48), 1f),
        )
        controls.addView(filterRow)

        val calibrationRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        calibrationRow.addView(text("Faktor kalibrasi:", 12f))
        val calibrationEdit = EditText(this).apply {
            setSingleLine(true)
            setText(calibrationFactor.format(3))
            setTextColor(0xffffffff.toInt())
            textSize = 14f
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or
                android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
            imeOptions = EditorInfo.IME_ACTION_DONE
        }
        calibrationRow.addView(
            calibrationEdit,
            LinearLayout.LayoutParams(dp(90), dp(48)),
        )
        val setCalibration = Button(this).apply {
            text = "Terapkan"
            setOnClickListener {
                val value = calibrationEdit.text.toString().toFloatOrNull()
                if (value == null || !value.isFinite() || value <= 0f) {
                    statusView.text = "Faktor kalibrasi harus berupa angka positif."
                } else {
                    calibrationFactor = value
                    tapPoint = null
                    tapDistanceMeters = null
                    statusView.text =
                        "Faktor ${value.format(3)} diterapkan; hanya valid untuk kamera validasi."
                }
            }
        }
        calibrationRow.addView(setCalibration)
        controls.addView(calibrationRow)

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

        overlayView.onMeasureTap = { x, y ->
            val result = latestResult
            val engine = inferenceEngine
            if (result == null || engine == null) {
                statusView.text = "Belum ada depth map untuk diukur."
            } else {
                val distance = engine.sampleTap(result, x, y, calibrationFactor)
                tapPoint = x to y
                tapDistanceMeters = distance
                statusView.text = distance?.let {
                    String.format(
                        Locale.US,
                        "Tap %.2f m / %.0f cm — estimasi, bukan alat ukur presisi.",
                        it,
                        it * 100f,
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
                    analysis.setAnalyzer(cameraExecutor) { image ->
                        var frame: Bitmap? = null
                        try {
                            val cameraFrame = CameraFrameConverter.toUprightBitmap(image)
                            frame = cameraFrame
                            val result = inferenceEngine?.runFrame(
                                bitmap = cameraFrame,
                                confidenceThreshold = confidenceThreshold,
                                selectedClasses = selectedClasses,
                                calibrationFactor = calibrationFactor,
                                fps = nextFps(),
                            )
                            if (result != null) {
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
                    provider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis,
                    )
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
        val decodeNote = if (config.detectorDecodeVerified) {
            "decode ditandai terverifikasi"
        } else {
            "decode/NMS BELUM diverifikasi PC"
        }
        statusView.text = String.format(
            Locale.US,
            "%.1f FPS · inferensi %.1f ms · %d objek · %s · %s",
            result.fps,
            result.inferenceLatencyMs,
            result.detections.size,
            result.runtimeLabel,
            decodeNote,
        )
    }

    private fun redrawLatest() {
        val result = latestResult ?: return
        overlayView.update(
            result,
            latestDepthBitmap,
            showDepthOverlay && !sideBySide,
            tapPoint,
            tapDistanceMeters,
        )
        depthMapView.setDepthBitmap(if (sideBySide) latestDepthBitmap else null)
    }

    private fun parseClasses(value: String): Set<String> =
        value.split(",")
            .map { it.trim().lowercase(Locale.ROOT) }
            .filter { it.isNotEmpty() }
            .toSet()

    private fun classFilterStatus(classes: Set<String>): String {
        if (classes.isEmpty()) return "Filter kelas dikosongkan; menampilkan semua kelas."
        val unknown = classes - CocoLabels.names.toSet()
        return if (unknown.isEmpty()) {
            "Filter aktif: ${classes.joinToString()}"
        } else {
            "Nama kelas COCO tidak dikenal: ${unknown.joinToString()}"
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
            android.util.Log.w(TAG, "Gagal menutup interpreter TFLite dengan bersih.", error)
        }
        if (::cameraExecutor.isInitialized) cameraExecutor.shutdown()
        latestDepthBitmap?.recycle()
    }

    companion object {
        private const val TAG = "Yolo26Depth"
    }
}
