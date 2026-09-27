// Deliberately plain and high-contrast: this app is shown to a room during
// a review from a few feet away, not used for daily typing, so large text
// and minimal chrome matter more than visual polish (per Phase B's
// requirement).
import 'package:flutter/material.dart';

class AppTheme {
  static const primary = Color(0xFF1652F0);
  static const danger = Color(0xFFD32F2F);
  static const success = Color(0xFF1B8A3E);

  static ThemeData get theme => ThemeData(
        useMaterial3: true,
        colorSchemeSeed: primary,
        scaffoldBackgroundColor: const Color(0xFFFAFAFA),
        appBarTheme: const AppBarTheme(
          centerTitle: true,
          elevation: 0,
          backgroundColor: Colors.white,
          foregroundColor: Colors.black,
          titleTextStyle: TextStyle(fontSize: 22, fontWeight: FontWeight.w700, color: Colors.black),
        ),
        textTheme: const TextTheme(
          headlineLarge: TextStyle(fontSize: 40, fontWeight: FontWeight.w800),
          headlineMedium: TextStyle(fontSize: 28, fontWeight: FontWeight.w700),
          titleLarge: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
          bodyLarge: TextStyle(fontSize: 18),
          bodyMedium: TextStyle(fontSize: 16),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            minimumSize: const Size.fromHeight(56),
            textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          ),
        ),
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        ),
      );
}
