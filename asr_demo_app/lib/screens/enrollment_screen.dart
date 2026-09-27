import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';

import '../services/api_client.dart';
import '../services/settings_service.dart';
import '../theme/app_theme.dart';
import '../widgets/record_button.dart';

/// Short, clear sentences to read for enrollment. These are NOT TORGO
/// prompts — Phase B explicitly says real TORGO prompts aren't needed for a
/// live demo, just something short and clear enough to read at a glance.
const _kEnrollmentPrompts = [
  'The quick brown fox jumps over the lazy dog.',
  'Please open the door and turn on the light.',
  'I would like a glass of water, thank you.',
];

enum _Stage { recording, submitting, training, done, failed }

class EnrollmentScreen extends StatefulWidget {
  const EnrollmentScreen({super.key});

  @override
  State<EnrollmentScreen> createState() => _EnrollmentScreenState();
}

class _EnrollmentScreenState extends State<EnrollmentScreen> {
  late final ApiClient _api;
  final _nameController = TextEditingController();
  final Map<int, File> _clips = {};

  _Stage _stage = _Stage.recording;
  String? _error;
  String? _jobId;
  Timer? _pollTimer;
  int? _epochsRun;
  double? _finalLoss;
  String? _note;

  @override
  void initState() {
    super.initState();
    _api = ApiClient(SettingsService());
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _nameController.dispose();
    super.dispose();
  }

  bool get _canSubmit => _nameController.text.trim().isNotEmpty && _clips.isNotEmpty;

  Future<void> _submit() async {
    setState(() {
      _stage = _Stage.submitting;
      _error = null;
    });
    try {
      final indices = _clips.keys.toList()..sort();
      final files = indices.map((i) => _clips[i]!).toList();
      final prompts = indices.map((i) => _kEnrollmentPrompts[i]).toList();
      final result = await _api.enrollStart(speakerId: _nameController.text.trim(), files: files, prompts: prompts);
      _jobId = result['job_id'] as String;
      setState(() => _stage = _Stage.training);
      _pollTimer = Timer.periodic(const Duration(seconds: 3), (_) => _poll());
      _poll();
    } on Object catch (e) {
      setState(() {
        _stage = _Stage.failed;
        _error = e.toString();
      });
    }
  }

  Future<void> _poll() async {
    if (_jobId == null) return;
    try {
      final status = await _api.enrollStatus(_jobId!);
      final s = status['status'] as String;
      if (s == 'running') return;
      _pollTimer?.cancel();
      setState(() {
        _stage = s == 'done' ? _Stage.done : _Stage.failed;
        _epochsRun = status['epochs_run'] as int?;
        _finalLoss = (status['final_loss'] as num?)?.toDouble();
        _note = status['note'] as String?;
        _error = status['error'] as String?;
      });
    } on Object catch (e) {
      // A transient network hiccup while polling shouldn't fail the whole
      // enrollment — the background job keeps running server-side
      // regardless; just try again on the next tick.
      setState(() => _error = 'Poll failed (will retry): $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Enroll a new speaker')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: switch (_stage) {
            _Stage.recording || _Stage.submitting => _recordingForm(),
            _Stage.training => _trainingProgress(),
            _Stage.done => _doneState(),
            _Stage.failed => _failedState(),
          },
        ),
      ),
    );
  }

  Widget _recordingForm() {
    final submitting = _stage == _Stage.submitting;
    return ListView(
      children: [
        TextField(
          controller: _nameController,
          enabled: !submitting,
          decoration: const InputDecoration(labelText: 'Speaker name / ID'),
          onChanged: (_) => setState(() {}),
        ),
        const SizedBox(height: 24),
        for (var i = 0; i < _kEnrollmentPrompts.length; i++) _promptCard(i),
        const SizedBox(height: 12),
        if (_error != null) Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: Text(_error!, style: const TextStyle(color: AppTheme.danger)),
        ),
        ElevatedButton(
          onPressed: (!submitting && _canSubmit) ? _submit : null,
          child: submitting
              ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
              : const Text('Start training'),
        ),
      ],
    );
  }

  Widget _promptCard(int i) {
    final recorded = _clips.containsKey(i);
    return Card(
      margin: const EdgeInsets.only(bottom: 14),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Please read:', style: TextStyle(color: Colors.black54, fontSize: 13)),
                  const SizedBox(height: 4),
                  Text(_kEnrollmentPrompts[i], style: Theme.of(context).textTheme.titleLarge),
                ],
              ),
            ),
            const SizedBox(width: 12),
            Column(
              children: [
                RecordButton(
                  size: 64,
                  idleLabel: recorded ? 'Re-record' : 'Record',
                  onRecorded: (f) => setState(() => _clips[i] = f),
                ),
                if (recorded) const Padding(
                  padding: EdgeInsets.only(top: 4),
                  child: Icon(Icons.check_circle, color: AppTheme.success),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _trainingProgress() {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircularProgressIndicator(),
          SizedBox(height: 24),
          Text('Training your personalized model...', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600)),
          SizedBox(height: 8),
          Text('This takes about a minute.', style: TextStyle(color: Colors.black54)),
        ],
      ),
    );
  }

  Widget _doneState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.check_circle, color: AppTheme.success, size: 64),
          const SizedBox(height: 16),
          Text('${_nameController.text.trim()} is ready!', style: Theme.of(context).textTheme.headlineMedium, textAlign: TextAlign.center),
          const SizedBox(height: 8),
          Text('$_epochsRun training steps · final loss ${_finalLoss?.toStringAsFixed(3) ?? "n/a"}', style: const TextStyle(color: Colors.black54)),
          if (_note != null) ...[
            const SizedBox(height: 12),
            Text(_note!, style: const TextStyle(color: Colors.orange), textAlign: TextAlign.center),
          ],
          const SizedBox(height: 24),
          ElevatedButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Back to Home')),
        ],
      ),
    );
  }

  Widget _failedState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.error, color: AppTheme.danger, size: 64),
          const SizedBox(height: 16),
          Text('Enrollment failed', style: Theme.of(context).textTheme.headlineMedium),
          const SizedBox(height: 8),
          Text(_error ?? 'Unknown error', style: const TextStyle(color: AppTheme.danger), textAlign: TextAlign.center),
          const SizedBox(height: 24),
          ElevatedButton(
            onPressed: () => setState(() {
              _stage = _Stage.recording;
              _error = null;
            }),
            child: const Text('Try again'),
          ),
        ],
      ),
    );
  }
}
