// Big tap-to-start/tap-to-stop record button, per Phase B's spec. Owns the
// `record` package's AudioRecorder lifecycle so both HomeScreen and
// EnrollmentScreen can drop in one of these without duplicating recorder
// setup/permission handling.
import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';

import '../theme/app_theme.dart';

class RecordButton extends StatefulWidget {
  /// Called with the recorded clip once the user stops recording.
  final ValueChanged<File> onRecorded;
  final String idleLabel;
  final double size;

  const RecordButton({
    super.key,
    required this.onRecorded,
    this.idleLabel = 'Tap to record',
    this.size = 120,
  });

  @override
  State<RecordButton> createState() => _RecordButtonState();
}

class _RecordButtonState extends State<RecordButton> {
  final _recorder = AudioRecorder();
  bool _isRecording = false;
  Duration _elapsed = Duration.zero;
  Timer? _ticker;
  String? _lastError;

  @override
  void dispose() {
    _ticker?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _toggle() async {
    if (_isRecording) {
      await _stop();
    } else {
      await _start();
    }
  }

  Future<void> _start() async {
    setState(() => _lastError = null);

    final micStatus = await Permission.microphone.request();
    if (!micStatus.isGranted) {
      setState(() => _lastError = 'Microphone permission is required to record.');
      return;
    }
    if (!await _recorder.hasPermission()) {
      setState(() => _lastError = 'Microphone permission is required to record.');
      return;
    }

    final dir = await getTemporaryDirectory();
    final path = '${dir.path}/clip_${DateTime.now().millisecondsSinceEpoch}.wav';

    // Record as 16kHz mono WAV directly — this exactly matches what the
    // backend/Whisper expects, so the common case never needs the server's
    // ffmpeg-conversion fallback (that fallback exists for robustness, not
    // as the normal path).
    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000, numChannels: 1),
      path: path,
    );

    setState(() {
      _isRecording = true;
      _elapsed = Duration.zero;
    });
    _ticker = Timer.periodic(const Duration(milliseconds: 200), (_) {
      setState(() => _elapsed += const Duration(milliseconds: 200));
    });
  }

  Future<void> _stop() async {
    _ticker?.cancel();
    final path = await _recorder.stop();
    setState(() => _isRecording = false);
    if (path == null) {
      setState(() => _lastError = 'Recording failed to save — please try again.');
      return;
    }
    widget.onRecorded(File(path));
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        GestureDetector(
          onTap: _toggle,
          child: Container(
            width: widget.size,
            height: widget.size,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _isRecording ? AppTheme.danger : AppTheme.primary,
              boxShadow: [
                BoxShadow(color: Colors.black.withValues(alpha: 0.15), blurRadius: 12, offset: const Offset(0, 4)),
              ],
            ),
            child: Icon(
              _isRecording ? Icons.stop_rounded : Icons.mic_rounded,
              color: Colors.white,
              size: widget.size * 0.45,
            ),
          ),
        ),
        const SizedBox(height: 10),
        Text(
          _isRecording ? '${_elapsed.inSeconds}s — tap to stop' : widget.idleLabel,
          style: Theme.of(context).textTheme.bodyLarge,
        ),
        if (_lastError != null) ...[
          const SizedBox(height: 6),
          Text(_lastError!, style: const TextStyle(color: AppTheme.danger), textAlign: TextAlign.center),
        ],
      ],
    );
  }
}
