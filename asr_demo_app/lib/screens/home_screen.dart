import 'dart:async';
import 'dart:io';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import '../models/profile.dart';
import '../services/api_client.dart';
import '../services/settings_service.dart';
import '../theme/app_theme.dart';
import '../widgets/record_button.dart';
import 'enrollment_screen.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late final ApiClient _api;

  List<Profile> _profiles = [Profile.generic('loading...')];
  Profile _selected = Profile.generic('loading...');

  bool _loadingProfiles = true;
  String? _healthError;

  bool _busy = false;
  String? _transcription;
  String? _modelUsed;
  double? _latencyMs;
  String? _transcribeError;

  final AudioPlayer _player = AudioPlayer();
  StreamSubscription<void>? _playerCompleteSub;
  bool _ttsLoading = false; // waiting on /speak
  bool _ttsPlaying = false;

  @override
  void initState() {
    super.initState();
    _api = ApiClient(SettingsService());
    _playerCompleteSub = _player.onPlayerComplete.listen((_) {
      if (mounted) setState(() => _ttsPlaying = false);
    });
    _refreshHealth();
  }

  @override
  void dispose() {
    _playerCompleteSub?.cancel();
    _player.dispose();
    super.dispose();
  }

  Future<void> _refreshHealth() async {
    setState(() {
      _loadingProfiles = true;
      _healthError = null;
    });
    try {
      final health = await _api.health();
      final poolLabel = (health['pooled_model']?['model_used_label'] as String?) ?? 'unknown';
      final personalized = (health['personalized_profiles'] as List<dynamic>? ?? [])
          .map((p) => Profile.personalized(p['speaker_id'] as String))
          .toList();
      final profiles = [Profile.generic(poolLabel), ...personalized];
      setState(() {
        _profiles = profiles;
        // Keep the current selection if it still exists (e.g. after
        // returning from enrollment), otherwise fall back to generic.
        _selected = profiles.firstWhere(
          (p) => p.speakerId == _selected.speakerId,
          orElse: () => profiles.first,
        );
        _loadingProfiles = false;
      });
    } on Object catch (e) {
      setState(() {
        _healthError = e.toString();
        _loadingProfiles = false;
      });
    }
  }

  Future<void> _onRecorded(File clip) async {
    await _player.stop(); // don't keep reading out the previous transcription
    setState(() {
      _ttsPlaying = false;
      _busy = true;
      _transcribeError = null;
      _transcription = null;
      _modelUsed = null;
      _latencyMs = null;
    });
    try {
      final result = await _api.transcribe(audioFile: clip, speakerId: _selected.speakerId);
      setState(() {
        _transcription = result['transcription'] as String? ?? '';
        _modelUsed = result['model_used'] as String? ?? '';
        _latencyMs = (result['latency_ms'] as num?)?.toDouble();
      });
    } on Object catch (e) {
      setState(() => _transcribeError = e.toString());
    } finally {
      setState(() => _busy = false);
      clip.delete().catchError((_) => clip); // best-effort cleanup of the local recording
    }
  }

  /// Fetches Piper speech for the current transcription from /speak and plays
  /// it; tapping again while it plays stops it. The WAV is written to a temp
  /// file and played as a DeviceFileSource rather than a BytesSource — on
  /// Android, file playback goes through MediaPlayer's most standard path.
  Future<void> _playback() async {
    final text = _transcription;
    if (text == null || _ttsLoading) return;
    if (_ttsPlaying) {
      await _player.stop();
      if (mounted) setState(() => _ttsPlaying = false);
      return;
    }

    setState(() => _ttsLoading = true);
    try {
      final wav = await _api.speak(text);
      final dir = await getTemporaryDirectory();
      final file = File('${dir.path}/tts_playback.wav');
      await file.writeAsBytes(wav, flush: true);
      await _player.stop();
      await _player.play(DeviceFileSource(file.path, mimeType: 'audio/wav'));
      if (mounted) setState(() => _ttsPlaying = true);
    } on TtsUnavailableException catch (e) {
      _showSnack('Text-to-speech is unavailable: ${e.message}');
    } on ApiException catch (e) {
      // e.g. a 400 "Text contains nothing speakable." for an empty/garbled transcription
      _showSnack("Couldn't play back: ${e.message}");
    } on Object catch (e) {
      _showSnack('Audio playback failed on this device: $e');
    } finally {
      if (mounted) setState(() => _ttsLoading = false);
    }
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg), duration: const Duration(seconds: 4)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ASR Demo'),
        actions: [
          IconButton(
            icon: const Icon(Icons.person_add_alt_1),
            tooltip: 'Enroll a new speaker',
            onPressed: () async {
              await Navigator.of(context).push(MaterialPageRoute(builder: (_) => const EnrollmentScreen()));
              _refreshHealth();
            },
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: 'Settings',
            onPressed: () async {
              await Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SettingsScreen()));
              _refreshHealth();
            },
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              if (_healthError != null) _unreachableBanner(),
              _profilePicker(),
              const SizedBox(height: 24),
              Expanded(child: Center(child: _resultArea())),
              const SizedBox(height: 16),
              RecordButton(onRecorded: _onRecorded, idleLabel: 'Tap to record and transcribe'),
              const SizedBox(height: 12),
            ],
          ),
        ),
      ),
    );
  }

  Widget _unreachableBanner() {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(color: AppTheme.danger.withValues(alpha: 0.1), borderRadius: BorderRadius.circular(10)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(_healthError!, style: const TextStyle(color: AppTheme.danger)),
          const SizedBox(height: 8),
          ElevatedButton(
            onPressed: _refreshHealth,
            style: ElevatedButton.styleFrom(minimumSize: const Size(0, 40), backgroundColor: AppTheme.danger),
            child: const Text('Retry'),
          ),
        ],
      ),
    );
  }

  Widget _profilePicker() {
    return Row(
      children: [
        const Icon(Icons.person_outline),
        const SizedBox(width: 8),
        Expanded(
          child: _loadingProfiles
              ? const LinearProgressIndicator()
              : DropdownButton<Profile>(
                  isExpanded: true,
                  value: _selected,
                  items: _profiles
                      .map((p) => DropdownMenuItem(value: p, child: Text(p.label, overflow: TextOverflow.ellipsis)))
                      .toList(),
                  onChanged: (p) => setState(() => _selected = p!),
                ),
        ),
      ],
    );
  }

  Widget _resultArea() {
    if (_busy) {
      return const Column(
        mainAxisSize: MainAxisSize.min,
        children: [CircularProgressIndicator(), SizedBox(height: 16), Text('Transcribing...')],
      );
    }
    if (_transcribeError != null) {
      return Text(_transcribeError!, style: const TextStyle(color: AppTheme.danger, fontSize: 16), textAlign: TextAlign.center);
    }
    if (_transcription == null) {
      return Text(
        'Record something to see it transcribed here.',
        style: Theme.of(context).textTheme.bodyLarge?.copyWith(color: Colors.black54),
        textAlign: TextAlign.center,
      );
    }
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(_transcription!, style: Theme.of(context).textTheme.headlineLarge, textAlign: TextAlign.center),
        const SizedBox(height: 12),
        Text('model: $_modelUsed  ·  ${_latencyMs?.toStringAsFixed(0)} ms', style: const TextStyle(color: Colors.black54)),
        const SizedBox(height: 16),
        OutlinedButton.icon(
          onPressed: _ttsLoading ? null : _playback,
          icon: _ttsLoading
              ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
              : Icon(_ttsPlaying ? Icons.stop : Icons.volume_up),
          label: Text(_ttsLoading ? 'Generating speech...' : (_ttsPlaying ? 'Stop playback' : 'Play back (TTS)')),
        ),
      ],
    );
  }
}
