import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class DiskBackedEmbeddingTests: XCTestCase {
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func testDenseEmbeddingReadsOnlyRequestedRowsAndRestoresShape() throws {
        let directory = try temporaryDirectory()
        let values = MLXArray(Array(0 ..< 48).map(Float.init), [6, 8]).asType(.float16)
        try save(
            arrays: [
                "language_model.model.embed_tokens_per_layer.weight": values,
                // Gemma 4 checkpoints colocate non-PLE tensors in these shards. The
                // disk reader must leave unsupported audio shapes/dtypes to the normal
                // model sanitizer instead of treating them as embedding rows.
                "audio_tower.encoder.layers.0.conv.weight": MLXArray.zeros(
                    [2, 2, 2], dtype: .uint8),
            ],
            url: directory.appendingPathComponent("model.safetensors"))

        let embedding = try DiskBackedEmbedding(
            modelDirectory: directory,
            weightSuffix: "embed_tokens_per_layer.weight",
            dimensions: 8,
            vocabularySize: 6)
        let result = embedding(MLXArray([4, 1, 4, 0]).reshaped(2, 2)).asType(.float32)

        XCTAssertEqual(result.shape, [2, 2, 8])
        XCTAssertEqual(
            result.asArray(Float.self),
            [
                32, 33, 34, 35, 36, 37, 38, 39,
                8, 9, 10, 11, 12, 13, 14, 15,
                32, 33, 34, 35, 36, 37, 38, 39,
                0, 1, 2, 3, 4, 5, 6, 7,
            ])
        XCTAssertEqual(embedding.shape.0, 6)
        XCTAssertEqual(embedding.shape.1, 8)
        XCTAssertEqual(embedding.weight.shape, [1, 1])
        XCTAssertTrue(
            embedding.checkpointTensorNames.contains(
                "audio_tower.encoder.layers.0.conv.weight"))
    }

    func testAffineQuantizedEmbeddingMatchesCheckpointDequantization() throws {
        let directory = try temporaryDirectory()
        let values = MLXArray(Array(0 ..< 256).map { Float($0) / 17 }, [4, 64])
        let (weight, scales, biases) = quantized(values, groupSize: 64, bits: 4)
        var checkpoint = [
            "language_model.model.embed_tokens_per_layer.weight": weight,
            "language_model.model.embed_tokens_per_layer.scales": scales,
        ]
        if let biases {
            checkpoint["language_model.model.embed_tokens_per_layer.biases"] = biases
        }
        try save(
            arrays: checkpoint,
            url: directory.appendingPathComponent("model.safetensors"))

        let embedding = try DiskBackedEmbedding(
            modelDirectory: directory,
            weightSuffix: "embed_tokens_per_layer.weight",
            dimensions: 64,
            vocabularySize: 4)
        let result = embedding(MLXArray([3, 0, 3])).asType(.float32)
        let expected = dequantized(
            weight[MLXArray([3, 0, 3])],
            scales: scales[MLXArray([3, 0, 3])],
            biases: biases?[MLXArray([3, 0, 3])],
            groupSize: 64,
            bits: 4
        ).asType(.float32)

        XCTAssertEqual(result.shape, [3, 64])
        XCTAssertEqual(result.asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual(
            embedding.storageTensorNames,
            Set(checkpoint.keys))
    }
}
