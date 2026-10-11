package id.ac.comvis.yolo26depth

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.MotionEvent
import android.view.View
import java.util.Locale

class DetectionOverlayView(context: Context) : View(context) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(255, 193, 7)
        style = Paint.Style.STROKE
        strokeWidth = 3.5f * resources.displayMetrics.density
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(88, 20, 20, 20)
        style = Paint.Style.FILL
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 13f * resources.displayMetrics.density
        style = Paint.Style.FILL
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }
    private val depthPaint = Paint(Paint.FILTER_BITMAP_FLAG).apply { alpha = 92 }
    private val tapPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 3f * resources.displayMetrics.density
    }

    private var frameWidth = 0
    private var frameHeight = 0
    private var detections: List<Detection> = emptyList()
    private var depthBitmap: Bitmap? = null
    private var showDepth = false
    private var tapPoint: Pair<Float, Float>? = null
    private var tapDistance: Float? = null
    var onMeasureTap: ((Float, Float) -> Unit)? = null

    init {
        isClickable = true
    }

    fun update(
        result: FrameResult,
        bitmap: Bitmap?,
        depthVisible: Boolean,
        measurePoint: Pair<Float, Float>?,
        measureMeters: Float?,
    ) {
        frameWidth = result.width
        frameHeight = result.height
        detections = result.detections
        depthBitmap = bitmap
        showDepth = depthVisible
        tapPoint = measurePoint
        tapDistance = measureMeters
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (frameWidth <= 0 || frameHeight <= 0) return
        val imageRect = fittedImageRect()
        val depth = depthBitmap
        if (showDepth && depth != null && !depth.isRecycled) {
            canvas.drawBitmap(depth, null, imageRect, depthPaint)
        }

        val scaleX = imageRect.width() / frameWidth
        val scaleY = imageRect.height() / frameHeight
        for (detection in detections) {
            val rect = RectF(
                imageRect.left + detection.left * scaleX,
                imageRect.top + detection.top * scaleY,
                imageRect.left + detection.right * scaleX,
                imageRect.top + detection.bottom * scaleY,
            )
            canvas.drawRect(rect, boxPaint)
            val distance = detection.distanceMeters?.let {
                String.format(Locale.US, "%.2f m / %.0f cm", it, it * 100f)
            } ?: "jarak tidak tersedia"
            val label = "${detection.className} - $distance"
            val textWidth = labelPaint.measureText(label)
            val labelHeight = labelPaint.textSize + 10f
            val labelTop = (rect.top - labelHeight).coerceAtLeast(imageRect.top)
            val labelRect = RectF(
                rect.left,
                labelTop,
                (rect.left + textWidth + 12f).coerceAtMost(imageRect.right),
                labelTop + labelHeight,
            )
            canvas.drawRect(labelRect, fillPaint)
            canvas.drawText(
                label,
                labelRect.left + 6f,
                labelRect.bottom - 5f,
                labelPaint,
            )
        }

        tapPoint?.let { point ->
            val x = imageRect.left + point.first * scaleX
            val y = imageRect.top + point.second * scaleY
            canvas.drawCircle(x, y, 12f * resources.displayMetrics.density, tapPaint)
            tapDistance?.let { distance ->
                canvas.drawText(
                    String.format(
                        Locale.US,
                        "Tap: %.2f m / %.0f cm",
                        distance,
                        distance * 100f,
                    ),
                    x + 14f,
                    y - 14f,
                    labelPaint,
                )
            }
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_UP || frameWidth <= 0 || frameHeight <= 0) {
            return true
        }
        val rect = fittedImageRect()
        if (!rect.contains(event.x, event.y)) return true
        val x = ((event.x - rect.left) * frameWidth / rect.width())
            .coerceIn(0f, frameWidth.toFloat())
        val y = ((event.y - rect.top) * frameHeight / rect.height())
            .coerceIn(0f, frameHeight.toFloat())
        onMeasureTap?.invoke(x, y)
        performClick()
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun fittedImageRect(): RectF {
        val scale = minOf(width.toFloat() / frameWidth, height.toFloat() / frameHeight)
        val fittedWidth = frameWidth * scale
        val fittedHeight = frameHeight * scale
        val left = (width - fittedWidth) / 2f
        val top = (height - fittedHeight) / 2f
        return RectF(left, top, left + fittedWidth, top + fittedHeight)
    }
}
