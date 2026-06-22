import AVFoundation
import UIKit

/// M1 engine: title card + chronological Ken Burns slideshow + (optional) music → exported .mp4.
/// Renders with AVVideoCompositionCoreAnimationTool (CALayers over a black base video).
///
/// TODO (house style, next milestones):
///  - M2: collage layouts (1/2/3/4-up, orientation-matched), slide/fade-in, varied transitions, vibrance (CIFilter).
///  - M3: on-device best-shot scoring (Vision aesthetics / faces) + one-per-day coverage.
///  This file is a starting scaffold — open in Xcode, build, and iterate.
enum ReelComposer {

    static let size = CGSize(width: 1080, height: 1920)
    static let fps: Int32 = 30
    static let perPhoto = 1.8          // seconds on screen
    static let titleLen = 2.8

    static func build(photos: [ReelPhoto], title: String, subtitle: String) async throws -> URL {
        guard !photos.isEmpty else { throw NSError(domain: "Reel", code: 1) }

        let total = titleLen + Double(photos.count) * perPhoto
        let base = try await makeBlackVideo(duration: total)   // base track for the animation tool

        // Parent + video layer for the CoreAnimation tool.
        let parent = CALayer(); parent.frame = CGRect(origin: .zero, size: size)
        let videoLayer = CALayer(); videoLayer.frame = parent.bounds
        parent.addSublayer(videoLayer)

        // Title cover over the first photo.
        addPhotoLayer(photos[0].image, to: parent, begin: 0, duration: titleLen, kenBurnsIn: true)
        addTitle(title, subtitle, to: parent, begin: 0, duration: titleLen)

        // Chronological slideshow.
        var t = titleLen
        for p in photos {
            addPhotoLayer(p.image, to: parent, begin: t, duration: perPhoto, kenBurnsIn: Bool.random())
            t += perPhoto
        }

        // Composition with the black base track + the animation tool.
        let comp = AVMutableComposition()
        let asset = AVURLAsset(url: base)
        let vTrack = comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)!
        if let src = try await asset.loadTracks(withMediaType: .video).first {
            try vTrack.insertTimeRange(CMTimeRange(start: .zero, duration: CMTime(seconds: total, preferredTimescale: 600)),
                                       of: src, at: .zero)
        }

        // Music: drop a track named "music.mp3" into the app bundle (loops/trims to length).
        if let musicURL = Bundle.main.url(forResource: "music", withExtension: "mp3") {
            let music = AVURLAsset(url: musicURL)
            if let aSrc = try await music.loadTracks(withMediaType: .audio).first {
                let aTrack = comp.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)!
                let mdur = try await music.load(.duration)
                let take = min(CMTimeGetSeconds(mdur), total)
                try aTrack.insertTimeRange(CMTimeRange(start: .zero, duration: CMTime(seconds: take, preferredTimescale: 600)),
                                           of: aSrc, at: .zero)
            }
        }

        let videoComp = AVMutableVideoComposition()
        videoComp.renderSize = size
        videoComp.frameDuration = CMTime(value: 1, timescale: fps)
        videoComp.animationTool = AVVideoCompositionCoreAnimationTool(postProcessingAsVideoLayer: videoLayer, in: parent)
        let instr = AVMutableVideoCompositionInstruction()
        instr.timeRange = CMTimeRange(start: .zero, duration: CMTime(seconds: total, preferredTimescale: 600))
        let layerInstr = AVMutableVideoCompositionLayerInstruction(assetTrack: vTrack)
        instr.layerInstructions = [layerInstr]
        videoComp.instructions = [instr]

        // Export.
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("reel-\(UUID().uuidString).mp4")
        guard let export = AVAssetExportSession(asset: comp, presetName: AVAssetExportPresetHighestQuality) else {
            throw NSError(domain: "Reel", code: 2)
        }
        export.outputURL = out
        export.outputFileType = .mp4
        export.videoComposition = videoComp
        await export.export()
        if export.status != .completed { throw export.error ?? NSError(domain: "Reel", code: 3) }
        return out
    }

    // MARK: - Layers

    private static func addPhotoLayer(_ image: UIImage, to parent: CALayer, begin: Double,
                                      duration: Double, kenBurnsIn: Bool) {
        guard let cg = image.cgImage else { return }
        let layer = CALayer()
        layer.frame = CGRect(origin: .zero, size: size)
        layer.contents = cg
        layer.contentsGravity = .resizeAspectFill   // cover-fill (TODO: top-bias crop / collages)
        layer.masksToBounds = true
        layer.opacity = 0

        // Crossfade in/out.
        let fade = CAKeyframeAnimation(keyPath: "opacity")
        fade.values = [0, 1, 1, 0]
        fade.keyTimes = [0, 0.12, 0.88, 1]
        fade.beginTime = begin == 0 ? AVCoreAnimationBeginTimeAtZero : begin
        fade.duration = duration
        fade.isRemovedOnCompletion = false
        layer.add(fade, forKey: "fade")

        // Ken Burns zoom.
        let zoom = CABasicAnimation(keyPath: "transform.scale")
        zoom.fromValue = kenBurnsIn ? 1.0 : 1.12
        zoom.toValue = kenBurnsIn ? 1.12 : 1.0
        zoom.beginTime = begin == 0 ? AVCoreAnimationBeginTimeAtZero : begin
        zoom.duration = duration
        zoom.isRemovedOnCompletion = false
        layer.add(zoom, forKey: "zoom")

        parent.addSublayer(layer)
    }

    private static func addTitle(_ title: String, _ subtitle: String, to parent: CALayer,
                                 begin: Double, duration: Double) {
        // Scrim for legibility.
        let scrim = CAGradientLayer()
        scrim.frame = CGRect(x: 0, y: size.height * 0.55, width: size.width, height: size.height * 0.45)
        scrim.colors = [UIColor.clear.cgColor, UIColor.black.withAlphaComponent(0.72).cgColor]
        parent.addSublayer(scrim)

        let t = CATextLayer()
        t.string = title
        t.font = UIFont.systemFont(ofSize: 84, weight: .bold)
        t.fontSize = 84
        t.alignmentMode = .center
        t.foregroundColor = UIColor.white.cgColor
        t.frame = CGRect(x: 0, y: size.height * 0.78, width: size.width, height: 110)
        t.contentsScale = 2
        parent.addSublayer(t)

        let s = CATextLayer()
        s.string = subtitle
        s.fontSize = 40
        s.alignmentMode = .center
        s.foregroundColor = UIColor.white.cgColor
        s.frame = CGRect(x: 0, y: size.height * 0.85, width: size.width, height: 60)
        s.contentsScale = 2
        parent.addSublayer(s)
        // TODO: fade title in/out (CAKeyframeAnimation on opacity) tied to [begin, begin+duration].
    }

    // MARK: - Black base video (gives the animation tool a track to render over)

    private static func makeBlackVideo(duration: Double) async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("base-\(UUID().uuidString).mp4")
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [AVVideoCodecKey: AVVideoCodecType.h264,
                                        AVVideoWidthKey: size.width, AVVideoHeightKey: size.height]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let attrs: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
                                     kCVPixelBufferWidthKey as String: size.width,
                                     kCVPixelBufferHeightKey as String: size.height]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attrs)
        writer.add(input)
        writer.startWriting(); writer.startSession(atSourceTime: .zero)

        let frames = Int(duration * Double(fps))
        var buf: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height), kCVPixelFormatType_32ARGB, attrs as CFDictionary, &buf)
        if let buf {  // a single black frame, presented for every frame time
            CVPixelBufferLockBaseAddress(buf, [])
            if let base = CVPixelBufferGetBaseAddress(buf) {
                memset(base, 0, CVPixelBufferGetBytesPerRow(buf) * Int(size.height))
            }
            CVPixelBufferUnlockBaseAddress(buf, [])
            for i in 0..<frames {
                while !input.isReadyForMoreMediaData { try? await Task.sleep(nanoseconds: 5_000_000) }
                adaptor.append(buf, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: fps))
            }
        }
        input.markAsFinished()
        await writer.finishWriting()
        return url
    }
}
