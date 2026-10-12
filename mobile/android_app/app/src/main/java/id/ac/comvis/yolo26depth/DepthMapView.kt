package id.ac.comvis.yolo26depth

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View
import java.util.Locale

class DepthMapView(context: Context) : View(context) {
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 255, 255)
        style = Paint.Style.STROKE
        strokeWidth = 2f * resources.displayMetrics.density
    }
    private val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(210, 0, 0, 0)
        style = Paint.Style.FILL
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 10f * resources.displayMetrics.density
        setShadowLayer(1f, 0f, 0f, Color.BLACK)
    }
    private var bitmap: Bitmap? = null
    private var frameWidth = 0
    private var frameHeight = 0
    private var detections: List<Detection> = emptyList()

    fun update(result: FrameResult?, value: Bitmap?) {
        bitmap = value
        frameWidth = result?.width ?: 0
        frameHeight = result?.height ?: 0
        detections = result?.detections.orEmpty()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(0xff101010.toInt())
        val source = bitmap ?: return
        if (source.isRecycled) return
        val imageRect = fittedImageRect(source)
        canvas.drawBitmap(source, null, imageRect, paint)
        if (frameWidth <= 0 || frameHeight <= 0) return

        val scaleX = imageRect.width() / frameWidth
        val scaleY = imageRect.height() / frameHeight
        for (detection in detections) {
            val box = RectF(
                imageRect.left + detection.left * scaleX,
                imageRect.top + detection.top * scaleY,
                imageRect.left + detection.right * scaleX,
                imageRect.top + detection.bottom * scaleY,
            )
            canvas.drawRect(box, boxPaint)

            val distance = detection.distanceMeters?.let {
                String.format(Locale.US, "%.2f m", it)
            } ?: "jarak -"
            val label = "${detection.className} · $distance"
            val maxLabelWidth = imageRect.width().coerceAtLeast(1f)
            val oldTextSize = labelPaint.textSize
            val textWidth = labelPaint.measureText(label)
            if (textWidth > maxLabelWidth) {
                labelPaint.textSize = oldTextSize * maxLabelWidth / textWidth
            }
            val labelWidth = labelPaint.measureText(label) + 8f
            val labelHeight = labelPaint.textSize + 8f
            val labelLeft = box.left.coerceIn(
                imageRect.left,
                (imageRect.right - labelWidth).coerceAtLeast(imageRect.left),
            )
            val labelTop = if (box.top - labelHeight >= imageRect.top) {
                box.top - labelHeight
            } else {
                box.top.coerceAtMost(imageRect.bottom - labelHeight)
            }
            canvas.drawRect(
                RectF(labelLeft, labelTop, labelLeft + labelWidth, labelTop + labelHeight),
                labelBackgroundPaint,
            )
            canvas.drawText(
                label,
                labelLeft + 4f,
                labelTop + labelPaint.textSize + 2f,
                labelPaint,
            )
            labelPaint.textSize = oldTextSize
        }
    }

    private fun fittedImageRect(source: Bitmap): RectF {
        val aspectWidth = if (frameWidth > 0) frameWidth else source.width
        val aspectHeight = if (frameHeight > 0) frameHeight else source.height
        val scale = minOf(width.toFloat() / aspectWidth, height.toFloat() / aspectHeight)
        val drawnWidth = aspectWidth * scale
        val drawnHeight = aspectHeight * scale
        val left = (width - drawnWidth) / 2f
        val top = (height - drawnHeight) / 2f
        return RectF(left, top, left + drawnWidth, top + drawnHeight)
    }
}
