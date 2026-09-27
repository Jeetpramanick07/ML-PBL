# asr_demo_app

Thin Flutter client for the `asr-personalization` live demo (Phase B of the
demo build plan). Talks to the FastAPI backend in
`../asr-personalization/src/api/main.py` over the local network — it does no
inference itself.

## Status

The `android/` platform folder is committed, so steps 2–3 of the setup below are already done. What's in place:
- **Manifest:** `RECORD_AUDIO` and `INTERNET` permissions, `usesCleartextTraffic="true"`, and the label "ASR Demo".
- **Toolchain it's tested with:** Flutter 3.47.x, Android platform 36, NDK 28.2 and **JDK 17**. Newer JDKs such as 25 are too new for the Android Gradle toolchain. Point Flutter at your SDK and JDK with `flutter config --android-sdk <path> --jdk-dir <path>`.
- **Dependencies:** `record` is `^7.1.1`. 5.x resolves to sub-packages that no longer compile together.
- **Build setting:** `android/gradle.properties` sets `kotlin.incremental=false`. When the pub cache and the project are on different Windows drives, Kotlin incremental compilation fails with "different roots".
- **Build:** `flutter build apk --release` → `build/app/outputs/flutter-apk/app-release.apk`. Install with `adb install -r build/app/outputs/flutter-apk/app-release.apk`.
- **Signing:** this is a **demo APK signed with the Flutter debug key**. It's fine for installing directly on a phone, and **not suitable for Play Store distribution**, which needs a release keystore and signing config.

A prebuilt APK is attached to the GitHub Release, not committed to git.

## Original setup notes

## Why this needs one manual setup step

The Flutter SDK isn't installed on the machine this app's code was written
on, so `flutter create` (which normally scaffolds the `android/`, `ios/`,
etc. platform folders) couldn't be run here. Everything under `lib/` and
`pubspec.yaml` is ready to go; you just need to generate the platform
folders once, on your machine, where Flutter *is* installed.

## One-time setup

1. Install the Flutter SDK if you haven't already, and confirm Android
   tooling is working:
   ```
   flutter doctor
   ```
   (Android Studio's SDK/command-line tools + an accepted license are
   enough — you don't need the Android Studio IDE itself.)

2. Scaffold the platform folders in a throwaway directory, then copy the
   `android/` folder into this project (this repo only targets Android —
   the demo device is a phone, not iOS):
   ```
   flutter create --org com.example --project-name asr_demo_app --platforms=android ../asr_demo_scaffold_tmp
   ```
   then copy `../asr_demo_scaffold_tmp/android` into this folder as
   `asr_demo_app/android` (this overwrites nothing of mine — `lib/` and
   `pubspec.yaml` here are untouched), and delete `asr_demo_scaffold_tmp`.

3. Edit `android/app/src/main/AndroidManifest.xml`:
   - Add these two lines as direct children of `<manifest>`, above
     `<application ...>`:
     ```xml
     <uses-permission android:name="android.permission.RECORD_AUDIO" />
     <uses-permission android:name="android.permission.INTERNET" />
     ```
   - Add `android:usesCleartextTraffic="true"` as an attribute on the
     `<application ...>` tag itself. **This one is easy to miss and will
     make every backend call silently fail on a real device** — Android 9+
     blocks plain HTTP by default, and the FastAPI backend is plain HTTP
     (no TLS) since it's a local-network-only demo server.
   - While you're in there, set `android:label="ASR Demo"` (or whatever
     name you want on the phone's home screen) on the same
     `<application ...>` tag — this is Phase C's "set a proper app name"
     step. A placeholder launcher icon (the default Flutter one) is fine
     for a PBL demo; if you want a custom one, drop a square PNG in
     `assets/icon.png`, add the `flutter_launcher_icons` dev dependency, and
     follow its README rather than hand-editing the `mipmap-*` folders.

4. Install dependencies:
   ```
   cd asr_demo_app
   flutter pub get
   ```
   If this (or the build in step 5) complains about `minSdkVersion`, raise
   `minSdkVersion` in `android/app/build.gradle` to at least `21` (the
   `record` package needs it) and re-run.

5. Connect your Android phone via USB with USB debugging enabled (Settings
   → Developer options), confirm it shows up:
   ```
   flutter devices
   ```
   then run the app on it:
   ```
   flutter run
   ```

## Using the app

1. Open **Settings** (gear icon) and enter the laptop's local IP and the
   backend's port, e.g. `192.168.1.42:8000` — find the laptop's IP with
   `ipconfig` (look for the WiFi adapter's IPv4 address) while
   `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` is running in
   `asr-personalization/`. Tap **Test connection** to confirm.
2. On **Home**, pick a profile (Generic model, or any personalized one
   already enrolled), tap the mic, speak, tap again to stop — the
   transcription, model used, and latency appear.
2. Tap the person-add icon to **enroll** a new speaker: enter a name, record
   the 1-3 short prompts shown on screen, tap **Start training**, and wait
   (~1 minute) for the personalized adapter to finish training server-side.
3. **Play back (TTS)** currently always shows "not available" — Piper isn't
   wired up in the backend yet (see `src/api/main.py`'s `/speak` docstring).
   This is expected, not a bug.

## Known limitations (by design, for a demo)

- No offline/on-device inference — if the backend is unreachable, the app
  shows an error and a retry button rather than doing anything itself.
- Only one enrollment fine-tune can run at a time on the backend.
- The enrollment training budget is capped (~75s server-side); very short
  clips may finish with an honestly-reported low-quality adapter rather
  than a polished one — see `note` in the enrollment success screen.

See `docs/DEMO_RUNBOOK.md` (Phase C) for the actual demo-day script and
pre-demo checklist once that's written.
