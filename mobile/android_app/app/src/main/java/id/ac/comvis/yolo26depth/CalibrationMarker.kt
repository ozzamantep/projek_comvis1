package id.ac.comvis.yolo26depth

import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.pdf.PdfDocument
import com.google.zxing.BarcodeFormat
import com.google.zxing.EncodeHintType
import com.google.zxing.qrcode.QRCodeWriter
import com.google.zxing.qrcode.decoder.ErrorCorrectionLevel
import java.io.OutputStream

object CalibrationMarker {
    const val PAYLOAD = "CVDEPTH10"
    const val SIDE_METERS = 0.10f
    private const val MODULES = 21
    private const val QUIET_ZONE_MM = 20.0
    private const val QR_SIDE_MM = 100.0
    private const val PAGE_SIDE_MM = QUIET_ZONE_MM * 2 + QR_SIDE_MM

    fun writePdf(output: OutputStream) {
        val matrix = QRCodeWriter().encode(
            PAYLOAD,
            BarcodeFormat.QR_CODE,
            MODULES,
            MODULES,
            mapOf(
                EncodeHintType.MARGIN to 0,
                EncodeHintType.ERROR_CORRECTION to ErrorCorrectionLevel.M,
                EncodeHintType.CHARACTER_SET to "UTF-8",
            ),
        )
        require(matrix.width == MODULES && matrix.height == MODULES) {
            "Marker QR tidak terbentuk dalam ukuran yang diharapkan."
        }

        val pagePoints = millimetersToPoints(PAGE_SIDE_MM).toInt()
        val quietZonePoints = millimetersToPoints(QUIET_ZONE_MM)
        val modulePoints = millimetersToPoints(QR_SIDE_MM) / MODULES
        val document = PdfDocument()
        try {
            val page = document.startPage(
                PdfDocument.PageInfo.Builder(pagePoints, pagePoints, 1).create(),
            )
            val paint = Paint().apply {
                color = Color.BLACK
                style = Paint.Style.FILL
                isAntiAlias = false
            }
            for (y in 0 until matrix.height) {
                for (x in 0 until matrix.width) {
                    if (matrix[x, y]) {
                        val left = quietZonePoints + x * modulePoints
                        val top = quietZonePoints + y * modulePoints
                        page.canvas.drawRect(
                            RectF(
                                left.toFloat(),
                                top.toFloat(),
                                (left + modulePoints).toFloat(),
                                (top + modulePoints).toFloat(),
                            ),
                            paint,
                        )
                    }
                }
            }
            document.finishPage(page)
            document.writeTo(output)
            output.flush()
        } finally {
            document.close()
        }
    }

    private fun millimetersToPoints(mm: Double): Double = mm * 72.0 / 25.4
}
