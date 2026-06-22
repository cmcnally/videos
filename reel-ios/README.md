# ReelMaker (iOS) — trip reels from iPhone photos

Goal: the trip-reel "house style" as a native iPhone app, so anyone can pick photos in the Photos
picker and get a finished reel — **no Claude, no API key, no Terminal.**

## Why native (not the Mac tool)
- **Photo picking**: `PhotosUI` `PhotosPicker` selects straight from the iPhone library.
- **Smart selection without Claude**: use **on-device** signals (Apple Vision aesthetics / face
  detection, `PHAsset` favorites & creation dates) instead of the Claude API.
- **Editing**: `AVFoundation` builds the video on-device — title cover, Ken Burns, collages,
  transitions, music — no ffmpeg.

## House style to port (from the Mac tool)
Title cover → chronological "a bit of every day" → varied collages (1/2/3/4-up, orientation-matched,
filled, white frames, slide/fade-in) → varied transitions → tasteful vibrance → music, beat-paced.

## Honest status & roadmap
This is a **starting scaffold**, not a finished app. A shippable app is a multi-step build.

- [x] Project plan + scaffold (this folder)
- [ ] **M1 – MVP**: pick photos → ordered-by-date slideshow (Ken Burns) + a music track + title card → export/share. (`ReelComposer` does this; needs Xcode build + iteration.)
- [ ] **M2 – Style**: collages (1/2/3/4-up), slide/fade-in, varied transitions, vibrance.
- [ ] **M3 – Smart selection**: on-device best-shot scoring (Vision/aesthetics), one-per-"day" coverage.
- [ ] **M4 – Polish + TestFlight**: settings (length, title text), then beta to testers.

## Build it
1. Open **Xcode** → File ▸ New ▸ **App** (SwiftUI, name "ReelMaker"). 
2. Add the `.swift` files in `ReelMaker/` to the target.
3. Add Info.plist usage string **NSPhotoLibraryUsageDescription** ("Pick photos to make a reel").
4. Run on a device/simulator.

## Get it to the testers (TestFlight)
1. Enroll in the **Apple Developer Program** ($99/yr).
2. In Xcode: set a Bundle ID + Team, Archive, upload to **App Store Connect**.
3. In App Store Connect ▸ **TestFlight**, add external testers by email (see `TESTERS.md`).
4. Testers install the free **TestFlight** app and tap your invite — no Claude, no setup.
