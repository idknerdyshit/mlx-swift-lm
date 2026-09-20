// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Regression coverage for mlx-community's 4-bit Gemma 4 vision checkpoint.
/// Its `embed_vision.embedding_projection` is stored as a quantized triplet,
/// so the module slot must accept `QuantizedLinear` during `loadWeights`.
struct Gemma4QuantizedVisionProjectionTests {
    private static let projectionPath = "embed_vision.embedding_projection"
    private static let groupSize = 32

    @Test("Gemma 4 loads a quantized multimodal embedding projection")
    func quantizedEmbeddingProjectionLoads() throws {
        let config = try Self.tinyConfiguration()
        let reference = Gemma4(config)
        let modules = Dictionary(uniqueKeysWithValues: reference.leafModules().flattened())
        let projection = try #require(modules[Self.projectionPath] as? Linear)

        var checkpoint = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        checkpoint.removeValue(forKey: "\(Self.projectionPath).weight")
        let (weight, scales, biases) = quantized(
            projection.weight, groupSize: Self.groupSize, bits: 4)
        checkpoint["\(Self.projectionPath).weight"] = weight
        checkpoint["\(Self.projectionPath).scales"] = scales
        if let biases {
            checkpoint["\(Self.projectionPath).biases"] = biases
        }

        let directory = URL(filePath: NSTemporaryDirectory())
            .appending(component: "gemma4-quantized-vision-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(arrays: checkpoint, url: directory.appending(component: "model.safetensors"))

        let model = Gemma4(config)
        let quantization = BaseConfiguration.PerLayerQuantization(
            perLayerQuantization: [
                Self.projectionPath: .quantize(
                    .init(groupSize: Self.groupSize, bits: 4))
            ])
        try loadWeights(
            modelDirectory: directory,
            model: model,
            perLayerQuantization: quantization)

        let loadedModules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let loadedProjection = try #require(
            loadedModules[Self.projectionPath] as? QuantizedLinear)
        #expect(loadedProjection.groupSize == Self.groupSize)
        #expect(loadedProjection.bits == 4)
        eval(model)
    }

    private static func tinyConfiguration() throws -> Gemma4Configuration {
        let json = """
            {
              "model_type": "gemma4",
              "image_token_id": 100,
              "text_config": {
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                "global_head_dim": 32, "vocab_size": 200, "vocab_size_per_layer_input": 200,
                "num_kv_shared_layers": 1, "hidden_size_per_layer_input": 8,
                "sliding_window": 16, "sliding_window_pattern": 2,
                "max_position_embeddings": 512
              },
              "vision_config": {
                "num_hidden_layers": 1, "hidden_size": 32, "intermediate_size": 64,
                "num_attention_heads": 2, "head_dim": 16, "patch_size": 4,
                "default_output_length": 4, "pooling_kernel_size": 1,
                "position_embedding_size": 8
              }
            }
            """
        return try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
    }
}
