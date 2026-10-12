package id.ac.comvis.yolo26depth

import org.junit.Assert.assertEquals
import org.junit.Test

class DepthLetterboxTest {
    @Test
    fun removesDepthLetterboxPadding() {
        val depth = FloatArray(4 * 4) { index -> (index % 4).toFloat() }
        val cropped = DepthMath.removeLetterbox(
            depth,
            modelWidth = 4,
            modelHeight = 4,
            transform = LetterboxTransform(
                scale = 1f,
                padX = 1f,
                padY = 0f,
                originalWidth = 2,
                originalHeight = 4,
            ),
        )

        assertEquals(2, cropped.width)
        assertEquals(4, cropped.height)
        assertEquals(listOf(1f, 2f, 1f, 2f, 1f, 2f, 1f, 2f), cropped.values.toList())
    }
}
