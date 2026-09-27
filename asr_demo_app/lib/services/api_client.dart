// Thin HTTP client for the Phase A FastAPI backend
// (asr-personalization/src/api/main.py). Every method throws an
// ApiException with a message safe to show directly in the UI — screens
// never need to interpret raw SocketException/TimeoutException/HTTP-status
// details themselves, so "backend unreachable" is handled in one place.
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'settings_service.dart';

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  @override
  String toString() => message;
}

/// Thrown specifically by [ApiClient.speak] when the backend's /speak
/// returns 503 (Piper voice missing/failed to load on the server) or 501
/// (an older backend without TTS) — distinct from a generic ApiException so
/// the UI can show a soft "not available" message instead of an error banner.
class TtsUnavailableException implements Exception {
  final String message;
  TtsUnavailableException(this.message);
}

class ApiClient {
  final SettingsService settings;
  static const _timeout = Duration(seconds: 30);
  // Enrollment fine-tuning is capped at ~75s server-side (see
  // src/api/enrollment.py); polling is a separate short-timeout GET, so this
  // long timeout only applies to /transcribe and /enroll/start themselves.

  ApiClient(this.settings);

  Future<Uri> _uri(String path) async {
    final hostPort = await settings.getHostPort();
    if (hostPort.isEmpty) {
      throw ApiException(
        'Backend address not set. Open Settings and enter your laptop\'s IP and port (e.g. 192.168.1.42:8000).',
      );
    }
    return Uri.parse('http://$hostPort$path');
  }

  Future<T> _guard<T>(Future<T> Function() body) async {
    try {
      return await body();
    } on TtsUnavailableException {
      rethrow;
    } on ApiException {
      rethrow;
    } on SocketException {
      throw ApiException('Cannot reach the backend. Check that it is running and that your phone is on the same WiFi as the laptop (or USB-tethered).');
    } on TimeoutException {
      throw ApiException('The backend took too long to respond. It may be busy (e.g. training an enrollment) — try again in a moment.');
    } on http.ClientException catch (e) {
      throw ApiException('Network error talking to the backend: ${e.message}');
    } on FormatException {
      throw ApiException('Backend returned an unexpected response. Is the address in Settings pointing at the right server?');
    }
  }

  Future<Map<String, dynamic>> health() => _guard(() async {
        final uri = await _uri('/health');
        final resp = await http.get(uri).timeout(_timeout);
        _checkStatus(resp);
        return jsonDecode(resp.body) as Map<String, dynamic>;
      });

  Future<Map<String, dynamic>> transcribe({required File audioFile, String? speakerId}) => _guard(() async {
        final uri = await _uri('/transcribe');
        final req = http.MultipartRequest('POST', uri);
        if (speakerId != null && speakerId.isNotEmpty) {
          req.fields['speaker_id'] = speakerId;
        }
        req.files.add(await http.MultipartFile.fromPath('file', audioFile.path));
        final streamed = await req.send().timeout(_timeout);
        final resp = await http.Response.fromStream(streamed);
        _checkStatus(resp);
        return jsonDecode(resp.body) as Map<String, dynamic>;
      });

  Future<Map<String, dynamic>> enrollStart({
    required String speakerId,
    required List<File> files,
    required List<String> prompts,
  }) =>
      _guard(() async {
        final uri = await _uri('/enroll/start');
        final req = http.MultipartRequest('POST', uri);
        req.fields['speaker_id'] = speakerId;
        // http's MultipartRequest.fields is a plain Map<String,String>, which
        // can't hold repeated keys the way FastAPI's `prompts: list[str] =
        // Form(...)` expects (one "prompts" part per clip). Adding each as a
        // filename-less MultipartFile part instead produces a plain
        // form-data field, and lets several share the same field name.
        for (final p in prompts) {
          req.files.add(http.MultipartFile.fromString('prompts', p));
        }
        for (final f in files) {
          req.files.add(await http.MultipartFile.fromPath('files', f.path));
        }
        final streamed = await req.send().timeout(_timeout);
        final resp = await http.Response.fromStream(streamed);
        _checkStatus(resp);
        return jsonDecode(resp.body) as Map<String, dynamic>;
      });

  Future<Map<String, dynamic>> enrollStatus(String jobId) => _guard(() async {
        final uri = await _uri('/enroll/status/$jobId');
        final resp = await http.get(uri).timeout(const Duration(seconds: 10));
        _checkStatus(resp);
        return jsonDecode(resp.body) as Map<String, dynamic>;
      });

  /// Returns the WAV file bytes for [text]. A 400 (empty/unspeakable/too
  /// long text) surfaces as an ApiException carrying the backend's message.
  Future<Uint8List> speak(String text) => _guard(() async {
        final uri = await _uri('/speak');
        final resp = await http.post(uri, body: {'text': text}).timeout(_timeout);
        if (resp.statusCode == 503 || resp.statusCode == 501) {
          final body = _tryDecodeDetail(resp.body);
          throw TtsUnavailableException(body ?? 'Text-to-speech is not available on the backend right now.');
        }
        _checkStatus(resp);
        final bytes = resp.bodyBytes;
        // Every WAV starts with "RIFF"; anything else means the address in
        // Settings points at something that isn't this backend.
        if (bytes.length < 44 || String.fromCharCodes(bytes.sublist(0, 4)) != 'RIFF') {
          throw ApiException('Backend returned something that is not WAV audio.');
        }
        return bytes;
      });

  String? _tryDecodeDetail(String body) {
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map && decoded['detail'] is String) return decoded['detail'] as String;
    } catch (_) {
      // fall through
    }
    return null;
  }

  void _checkStatus(http.Response resp) {
    if (resp.statusCode >= 200 && resp.statusCode < 300) return;
    final detail = _tryDecodeDetail(resp.body);
    throw ApiException(detail ?? 'Backend returned HTTP ${resp.statusCode}.');
  }
}
