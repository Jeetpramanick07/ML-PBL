import 'package:flutter/material.dart';

import '../services/api_client.dart';
import '../services/settings_service.dart';
import '../theme/app_theme.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _settings = SettingsService();
  late final ApiClient _api;
  final _controller = TextEditingController();

  bool _testing = false;
  String? _testResult;
  bool _testOk = false;

  @override
  void initState() {
    super.initState();
    _api = ApiClient(_settings);
    _load();
  }

  Future<void> _load() async {
    _controller.text = await _settings.getHostPort();
    setState(() {});
  }

  Future<void> _save() async {
    await _settings.setHostPort(_controller.text.trim());
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Saved.')));
    }
  }

  Future<void> _testConnection() async {
    await _save();
    setState(() {
      _testing = true;
      _testResult = null;
    });
    try {
      final health = await _api.health();
      final device = health['device'];
      final pooled = health['pooled_model']?['model_used_label'];
      setState(() {
        _testOk = true;
        _testResult = 'Connected. Backend device: $device, pooled model: $pooled.';
      });
    } on Object catch (e) {
      setState(() {
        _testOk = false;
        _testResult = e.toString();
      });
    } finally {
      setState(() => _testing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Backend address',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 6),
              const Text(
                'The laptop\'s local IP and port, e.g. 192.168.1.42:8000. '
                'This changes with the WiFi network, so re-enter it on demo day.',
                style: TextStyle(color: Colors.black54),
              ),
              const SizedBox(height: 14),
              TextField(
                controller: _controller,
                decoration: const InputDecoration(labelText: 'host:port', hintText: '192.168.1.42:8000'),
                keyboardType: TextInputType.url,
              ),
              const SizedBox(height: 20),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: _save,
                      child: const Text('Save'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: _testing ? null : _testConnection,
                      child: _testing
                          ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                          : const Text('Test connection'),
                    ),
                  ),
                ],
              ),
              if (_testResult != null) ...[
                const SizedBox(height: 16),
                Text(_testResult!, style: TextStyle(color: _testOk ? AppTheme.success : AppTheme.danger)),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
