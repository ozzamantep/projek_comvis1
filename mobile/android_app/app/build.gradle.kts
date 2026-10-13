plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val multitaskModel = rootProject.projectDir
    .resolve(
        "../../yolo26n_multitask_coco4_v1_20261013-163528-539afa/" +
            "inference/multitask.onnx"
    )
val generatedModelAssets = layout.buildDirectory.dir("generated/multitaskModelAssets")
val syncMultitaskModel = tasks.register<Sync>("syncMultitaskModel") {
    from(multitaskModel)
    into(generatedModelAssets)
    doFirst {
        require(multitaskModel.isFile) {
            "Combined ONNX model is missing: $multitaskModel. " +
                "Restore the new training export in its inference folder."
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
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
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
    dependsOn(syncMultitaskModel)
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.0")
    implementation("com.google.mlkit:barcode-scanning:17.3.0")
    implementation("com.google.zxing:core:3.5.3")
    testImplementation("junit:junit:4.13.2")
}
