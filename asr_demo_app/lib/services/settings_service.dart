// Persists the backend's host:port locally. This changes every time the
// demo runs on a different WiFi network, so it must never be hardcoded
// (see Phase B's requirement) — the phone and laptop are different devices,
// so 127.0.0.1/localhost from the phone would point at the phone itself.
import 'package:shared_preferences/shared_preferences.dart';

class SettingsService {
  static const _hostKey = 'backend_host_port';
  static const defaultHostPort = '192.168.1.100:8000'; // placeholder only — always shown as unset in the UI

  Future<String> getHostPort() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_hostKey) ?? '';
  }

  Future<void> setHostPort(String hostPort) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_hostKey, hostPort.trim());
  }
}
