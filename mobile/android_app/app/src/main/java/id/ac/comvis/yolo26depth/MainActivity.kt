package id.ac.comvis.yolo26depth

import android.Manifest
import android.content.Intent
import android.content.res.Configuration
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.PointF
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.Surface
import android.view.View
import android.view.inputmethod.EditorInfo
import android.hardware.camera2.CameraCharacteristics
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
import androidx.camera.core.UseCaseGroup
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.barcode.BarcodeScanner
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import java.util.Locale
import java.io.IOException
import java.io.OutputStreamWriter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.hypot
import kotlin.math.min

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
