/// A selectable speaker profile in the Home screen's dropdown, built from
/// GET /health. `speakerId == null` means "no personalized adapter" — the
/// backend then uses its pooled LoRA model (or zero-shot, if no pooled
/// adapter was found either; /health's [poolModelLabel] says which).
class Profile {
  final String? speakerId;
  final String label;

  const Profile({required this.speakerId, required this.label});

  static Profile generic(String poolModelLabel) =>
      Profile(speakerId: null, label: 'Generic model ($poolModelLabel)');

  factory Profile.personalized(String speakerId) =>
      Profile(speakerId: speakerId, label: 'Personalized: $speakerId');

  // DropdownButton matches its `value` against `items` by ==, and every
  // refresh rebuilds a fresh list of Profile instances — without value
  // equality (identity is the default), the freshly-selected Profile would
  // never == any entry in the new items list and Flutter would throw a
  // "there should be exactly one item with [DropdownButton]'s value"
  // assertion the moment /health resolves.
  @override
  bool operator ==(Object other) => other is Profile && other.speakerId == speakerId;

  @override
  int get hashCode => speakerId.hashCode;
}
