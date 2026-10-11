package id.ac.comvis.yolo26depth

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.view.View

class DepthMapView(context: Context) : View(context) {
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private var bitmap: Bitmap? = null

    fun setDepthBitmap(value: Bitmap?) {
        bitmap = value
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(0xff101010.toInt())
        val source = bitmap ?: return
        if (source.isRecycled) return
        val scale = minOf(width.toFloat() / source.width, height.toFloat() / source.height)
        val drawnWidth = source.width * scale
        val drawnHeight = source.height * scale
        val left = (width - drawnWidth) / 2f
        val top = (height - drawnHeight) / 2f
        canvas.drawBitmap(
            source,
            null,
            android.graphics.RectF(left, top, left + drawnWidth, top + drawnHeight),
            paint,
        )
    }
}
