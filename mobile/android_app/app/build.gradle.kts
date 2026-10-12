plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val calibratedDepthModel = rootProject.projectDir
    .resolve("../../data/yolo26n-depth-nyu-calibrated.tflite")
val generatedModelAssets = layout.buildDirectory.dir("generated/calibratedDepthAssets")
val syncCalibratedDepthModel = tasks.register<Copy>("syncCalibratedDepthModel") {
    from(calibratedDepthModel)
    into(generatedModelAssets)
    doFirst {
        require(calibratedDepthModel.isFile) {
            "Depth model is missing: $calibratedDepthModel. " +
                "Place yolo26n-depth-nyu-calibrated.tflite in the repository data folder."
        }
    }
}

android {
    namespace = "id.ac.comvis.yolo26depth"
    compileSdk = 35

    defaultConfig {
        applicationId = "id.ac.comvis.yolo26depth"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
    }

    sourceSets.getByName("main").assets.srcDir(generatedModelAssets)

    buildFeatures {
        buildConfig = false
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

tasks.named("preBuild").configure {
    dependsOn(syncCalibratedDepthModel)
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")
    testImplementation("junit:junit:4.13.2")
}
