import 'dart:io';

import 'package:flutter/material.dart';

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

  @override
  void initState() {
    super.initState();
    _api = ApiClient(SettingsService());
    _refreshHealth();
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
    setState(() {
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

  Future<void> _playback() async {
    if (_transcription == null) return;
    try {
      await _api.speak(_transcription!);
      // Actual audio playback would need an audio-player package; out of
      // scope here since Piper isn't wired up server-side yet (this call
      // will currently always raise TtsUnavailableException below).
    } on TtsUnavailableException catch (e) {
      _showSnack(e.message);
    } on Object catch (e) {
      _showSnack(e.toString());
    }
  }

  void _showSnack(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
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
          onPressed: _playback,
          icon: const Icon(Icons.volume_up),
          label: const Text('Play back (TTS)'),
        ),
      ],
    );
  }
}
