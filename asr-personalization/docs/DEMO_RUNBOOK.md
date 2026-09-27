# Demo Runbook — ASR Personalization live demo

One-page reference for running the live demo (backend on laptop + Flutter
app on phone). See `src/api/` (Phase A) and `../asr_demo_app/` (Phase B) for
the actual implementation; this file is the "what to actually do" script.

## 1. Start the backend (do this before anyone arrives)

From `asr-personalization/`, with the project's venv active:

```
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Leave this terminal window open and visible for the whole demo — if
something goes wrong, the logs printed here (and in `reports/api_server.log`
/ `reports/api_requests.log`) are the fastest way to see why.

Wait for the startup log line `Ready. LAN IP appears to be ...` — it prints
its best guess at the laptop's IP, but confirm it yourself too (see below),
since guesses from `socket.gethostname()` are occasionally wrong on
multi-adapter laptops.

**Find the laptop's IP to type into the phone's Settings screen:**

- Windows: open a new terminal and run `ipconfig`, find the **Wireless LAN
  adapter Wi-Fi** section, and read its `IPv4 Address` (e.g. `192.168.1.42`).
  Ignore any `192.168.56.x` or `172.x` addresses — those are usually
  VirtualBox/WSL/VPN virtual adapters, not the real WiFi one.
- Confirm the backend is reachable from a browser on your own laptop first:
  `http://<that IP>:8000/health` should return JSON, not an error.

## 2. Demo script

### What's loaded
- **Pooled model:** `model_used = pooled-lora:generic_full`. This is Whisper-small plus the final pooled LoRA adapter (epoch 6, val WER 16.30%).
- **Personalized profiles:** F03 (mild), M04 (severe) and M05 (moderate). These are real per-speaker adapters from the master run, each trained on 5 minutes of that TORGO speaker's audio. They show up as `personalized:F03` and so on.

Source for every number below: `reports/master_results_final.md`.

### Talking points (the real numbers, use these)
1. **The problem.** Off-the-shelf Whisper-small gets **57.52% WER on dysarthric speech**. That's measured on the clean subset of our held-out test set (1,363 dysarthric utterances). More than half the words are wrong. On healthy speech it's about 11%.
2. **Pooled adaptation.** After fine-tuning a small LoRA adapter (0.36% of the model's parameters) on pooled TORGO speech, dysarthric WER drops to **31.55%**. That's a **45.1% relative reduction**. Healthy-speech WER also improves, from 10.9% to 4.7%, so the adaptation doesn't hurt typical speakers.
3. **Personalization.** Adding a per-speaker adapter trained on just **5 minutes** of that speaker's audio gave further WER gains for **2 of the 3 speakers tested**:
   - M05 (moderate): 29.1% → 23.5%
   - M04 (severe): 56.1% → 52.8%, though its character error rate got worse
   - F03 (mild): 24.2% → 23.5%, which is within noise

   Say plainly: *"This is from about 170 test utterances per speaker and a single training run. It's promising, not a definitive result."*
4. **Honest scope.** The pooled model was trained on a 40% subsample, and training was stopped by a 4.5 h compute cap after 7 epochs, while it was still improving. These numbers are a lower bound.

### If a reviewer asks about the evaluation methodology
Answer directly. Don't deflect.
- *"Is this speaker-independent?"* **No.** Our splits are utterance-level. The test utterances we report were never seen in training, but the model **has heard these speakers' voices** on other utterances. So this is an **utterance-unseen, not a fully speaker-independent** result, and speaker-independent performance would likely be worse.
- *"Were test clips in the training data?"* Some rows in our speaker-held-out split turned out to overlap with the training subsample (32%). We caught this and excluded them. The numbers above use only the **clean subset**, with no audio seen in training or validation. The full-split figure (23.3% dysarthric WER) is inflated by that overlap, so we don't quote it.
- *"How would you fix that?"* Retrain the pooled model with the test speakers fully excluded. We did this at small scale (`generic_unseen`, 1,200 rows), but not at the final compute budget. That's the next step.

### Live flow
1. **Generic model first.** On the phone, leave the profile dropdown on "Generic model (pooled-lora:generic_full)". Speak a short phrase, then show the transcription, the `model: pooled-lora:generic_full` line, and the latency.
2. **Personalized profiles: demonstrate them with the speakers' own audio.** F03, M04 and M05 are adapters for specific TORGO speakers. Selecting "Personalized: M04" and speaking in *your* voice doesn't demonstrate anything meaningful, because the adapter is fit to M04's speech, not yours. Instead, play one of that speaker's held-out clips and show both outputs. These are verified examples: the audio was never used in training or validation, and the API output matches the offline evaluation. Run from `asr-personalization/` on the laptop:
   ```
   curl -F "file=@data/raw/torgo/M04/Session2/wav_headMic/0092.wav" http://127.0.0.1:8000/transcribe
   curl -F "file=@data/raw/torgo/M04/Session2/wav_headMic/0092.wav" -F "speaker_id=M04" http://127.0.0.1:8000/transcribe
   ```
   - Reference text: "gadget". Pooled model: **"gatish"**. M04 profile: **"gadget"**.
   - Others: F03 `data/raw/torgo/F03/Session1/wav_arrayMic/0181.wav` (ref "sleep": pooled "roop", F03 "loop").
   - The interactive `http://127.0.0.1:8000/docs` page does the same thing with a file picker, if a GUI reads better on the projector.
   - Pick examples honestly. The profiles don't fix every clip (see the numbers above). If asked, say these were chosen to illustrate the mechanism, and the aggregate WER numbers are the evidence.
   - The app's profile dropdown still shows these profiles, and switching changes the `model:` line to `personalized:M04`. That's a fine way to show the *switching mechanism*, as long as you say it's not your voice's adapter.
3. **Live enrollment, if time and confidence allow.** Tap the person-add
   icon, enter a name, read the three on-screen prompts, tap "Start
   training", and talk through what's happening while it trains
   (~1 minute): "this is fitting a small adapter on top of the pooled model,
   using only the three clips just recorded." Once done, switch to that new
   profile and transcribe something.

Keep step 3 optional and clearly framed as a bonus — if it's slow, or the
adapter from only 1-3 tiny clips doesn't sound obviously better (a real and
expected possibility, noted honestly in the app's own enrollment success
screen), that is not a demo failure. The report's actual quantitative
personalization results (from properly-sized enrollment sets, not a live
30-second clip) are the real evidence; this flow demonstrates the mechanism
working end-to-end, not a guaranteed live WER improvement.

## 3. If WiFi fails

Switch to USB tethering as a fallback network that doesn't depend on venue
WiFi:

1. Connect the phone to the laptop via USB.
2. On the phone: Settings → Network & internet → Hotspot & tethering → USB
   tethering → on. (Exact menu path varies by Android version/vendor.)
3. Run `ipconfig` on the laptop again — a new adapter (something like
   "Ethernet adapter ... (Remote NDIS...)" or similar) will show a new IPv4
   address, usually `192.168.42.x` or similar Android-tethering range.
4. Re-enter that new IP in the phone app's Settings screen and hit "Test
   connection" again.

Rehearse this switch once before the actual review — it's a couple of menu
taps, not a code change, but not the kind of thing you want to be figuring
out live in front of an audience for the first time.

## 4. Pre-demo checklist (do this the day before, not the morning of)

- [ ] Backend starts cleanly (`uvicorn ...`). In `/health`, `pooled_model.source_checkpoint` should read `checkpoints/generic_full/best` and `model_used_label` should read `pooled-lora:generic_full`. Any other checkpoint means `generic_full/best` is missing and it fell back to an early test model; `null` means it fell back to zero-shot. Check `reports/api_server.log` in either case.
- [ ] `/health`'s `personalized_profiles` lists **F03, M04, M05** with `"source": "pretrained"`. If they're missing, `generic_full/best` didn't load (they're only registered on the base they were trained on), or `checkpoints/personalized_final/` is missing.
- [ ] Both `curl` comparisons in the demo script (M04 "gadget", F03 "sleep") give the outputs listed there.
- [ ] Phone has the app installed, is connected to the same WiFi as the laptop, and Settings → Test connection succeeds.
- [ ] USB tethering fallback has been tested at least once end-to-end
      (Section 3), not just read about.
- [ ] One full dry run of the exact demo script above, on the actual venue
      WiFi if at all possible (or the closest approximation available).
