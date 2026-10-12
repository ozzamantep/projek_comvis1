package id.ac.comvis.yolo26depth

import org.junit.Assert.assertEquals
import org.junit.Test

class DepthCalibrationTest {
    @Test
    fun fitsLinearCalibrationFromPairs() {
        val calibration = LinearDepthCalibration.fit(
            listOf(
                CalibrationPair(actualMeters = 1f, modelMeters = 0.5f),
                CalibrationPair(actualMeters = 3f, modelMeters = 1.5f),
                CalibrationPair(actualMeters = 5f, modelMeters = 2.5f),
            ),
        )

        assertEquals(2f, calibration.a, 1e-5f)
        assertEquals(0f, calibration.b, 1e-5f)
        assertEquals(4f, calibration.apply(2f), 1e-5f)
    }

    @Test
    fun parsesLineAndSemicolonSeparatedPairs() {
        val pairs = LinearDepthCalibration.parsePairs("1.0,0.5; 3.0,1.5\n5.0,2.5")

        assertEquals(3, pairs.size)
        assertEquals(3f, pairs[1].actualMeters, 1e-5f)
        assertEquals(1.5f, pairs[1].modelMeters, 1e-5f)
    }

    @Test
    fun patchSamplingUsesMedianInsteadOfCenterPixel() {
        val depth = FloatArray(21 * 21) { 2f }
        depth[10 * 21 + 10] = 0.2f

        assertEquals(2f, DepthMath.sampleMedian(depth, 21, 21, 10f, 10f, 15)!!, 1e-5f)
    }

    @Test
    fun tapHistoryUsesMedianAcrossSevenFrames() {
        val history = DepthFrameHistory(maxFrames = 7)
        for (value in 1..7) history.add(FloatArray(15 * 15) { value.toFloat() }, 15, 15)

        assertEquals(4f, history.sampleMedian(7f, 7f, 15)!!, 1e-5f)
    }

    @Test
    fun objectDistanceUsesSevenFrameMedianForSameTrack() {
        val smoother = DetectionDistanceSmoother(maxFrames = 7)
        var smoothed: Detection? = null
        for (value in 1..7) {
            smoothed = smoother.smooth(
                listOf(
                    Detection(
                        classId = 0,
                        className = "person",
                        confidence = 0.9f,
                        left = 10f,
                        top = 10f,
                        right = 40f,
                        bottom = 60f,
                        rawDistanceMeters = value.toFloat(),
                        distanceMeters = value.toFloat(),
                        rawCenterX = 0.5f,
                        rawCenterY = 0.5f,
                        rawWidth = 0.2f,
                        rawHeight = 0.2f,
                    ),
                ),
                LinearDepthCalibration.IDENTITY,
            ).single()
        }

        assertEquals(4f, smoothed!!.rawDistanceMeters!!, 1e-5f)
        assertEquals(4f, smoothed.distanceMeters!!, 1e-5f)
    }
}
