import SwiftUI
import PhotosUI
import AVKit

/// MVP screen: pick photos → make a reel → preview & share.
struct ContentView: View {
    @State private var picks: [PhotosPickerItem] = []
    @State private var photos: [ReelPhoto] = []
    @State private var title = "MY TRIP"
    @State private var subtitle = "2026"
    @State private var building = false
    @State private var reelURL: URL?
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("Cover text") {
                    TextField("Title", text: $title)
                    TextField("Subtitle", text: $subtitle)
                }
                Section("Photos") {
                    PhotosPicker(selection: $picks, maxSelectionCount: 80, matching: .images) {
                        Label(photos.isEmpty ? "Pick photos" : "\(photos.count) selected", systemImage: "photo.stack")
                    }
                }
                Section {
                    Button {
                        Task { await makeReel() }
                    } label: {
                        if building { ProgressView() } else { Text("Make Reel").bold() }
                    }
                    .disabled(photos.isEmpty || building)
                }
                if let reelURL {
                    Section("Result") {
                        VideoPlayer(player: AVPlayer(url: reelURL)).frame(height: 380)
                        ShareLink("Share / Save", item: reelURL)
                    }
                }
                if let error { Text(error).foregroundStyle(.red) }
            }
            .navigationTitle("ReelMaker")
            .onChange(of: picks) { _, _ in Task { await loadPhotos() } }
        }
    }

    /// Load picked images + their capture dates (for chronological order).
    private func loadPhotos() async {
        var loaded: [ReelPhoto] = []
        for item in picks {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else { continue }
            // Capture date: prefer the asset's creationDate via the local identifier.
            var date = Date()
            if let id = item.itemIdentifier {
                let assets = PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil)
                if let d = assets.firstObject?.creationDate { date = d }
            }
            loaded.append(ReelPhoto(image: image, date: date))
        }
        photos = loaded.sorted { $0.date < $1.date }   // chronological
    }

    private func makeReel() async {
        building = true; error = nil
        defer { building = false }
        do {
            reelURL = try await ReelComposer.build(photos: photos, title: title, subtitle: subtitle)
        } catch {
            self.error = "Couldn't build reel: \(error.localizedDescription)"
        }
    }
}

struct ReelPhoto: Identifiable {
    let id = UUID()
    let image: UIImage
    let date: Date
}
