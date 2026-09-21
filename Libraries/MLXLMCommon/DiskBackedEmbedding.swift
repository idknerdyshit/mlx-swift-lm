// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A model-specific replacement performed before checkpoint tensors are materialized.
///
/// The loader omits `excludedTensorNames`, merges `placeholderWeights`, and then follows its
/// normal sanitize / quantize / strict-update path. The placeholder keeps strict verification
/// intact for the tiny parameter owned by the replacement module.
package struct DiskBackedWeightPlan {
    package let excludedTensorNames: Set<String>
    package let placeholderWeights: [String: MLXArray]

    package init(
        excludedTensorNames: Set<String>,
        placeholderWeights: [String: MLXArray]
    ) {
        self.excludedTensorNames = excludedTensorNames
        self.placeholderWeights = placeholderWeights
    }
}

/// Opt-in hook for models that can leave selected checkpoint tensors on disk.
package protocol DiskBackedWeightsProviding {
    func prepareDiskBackedWeights(in modelDirectory: URL) throws -> DiskBackedWeightPlan?
}

package enum DiskBackedEmbeddingError: LocalizedError {
    case missingTensor(String)
    case ambiguousTensor(String, [String])
    case malformedSafetensors(URL)
    case unsupportedDType(String)
    case invalidShape(String, [Int])
    case incompatibleQuantization(String)
    case tokenOutOfRange(Int, Int)

    package var errorDescription: String? {
        switch self {
        case .missingTensor(let suffix):
            "No safetensors entry ends in \(suffix)."
        case .ambiguousTensor(let suffix, let names):
            "Multiple safetensors entries end in \(suffix): \(names.joined(separator: ", "))."
        case .malformedSafetensors(let url):
            "Malformed safetensors file at \(url.path)."
        case .unsupportedDType(let dtype):
            "Unsupported disk-backed embedding dtype \(dtype)."
        case .invalidShape(let name, let shape):
            "Invalid disk-backed embedding shape for \(name): \(shape)."
        case .incompatibleQuantization(let message):
            message
        case .tokenOutOfRange(let token, let vocabularySize):
            "Embedding token \(token) is outside 0..<\(vocabularySize)."
        }
    }
}

private struct SafetensorEntry {
    let name: String
    let dtypeName: String
    let dtype: DType
    let shape: [Int]
    let data: Data
    let dataStart: Int
    let byteCount: Int

    var rowCount: Int { shape[0] }
    var rowWidth: Int { shape[1] }
    var rowByteCount: Int { byteCount / rowCount }

    func rows(_ indices: [Int]) throws -> MLXArray {
        var bytes = Data(capacity: indices.count * rowByteCount)
        for index in indices {
            guard index >= 0, index < rowCount else {
                throw DiskBackedEmbeddingError.tokenOutOfRange(index, rowCount)
            }
            let start = dataStart + index * rowByteCount
            bytes.append(data[start ..< start + rowByteCount])
        }
        return MLXArray(bytes, [indices.count, rowWidth], dtype: dtype)
    }
}

private func safetensorDType(_ name: String) throws -> DType {
    switch name {
    case "BF16": .bfloat16
    case "F16": .float16
    case "F32": .float32
    case "U32": .uint32
    default: throw DiskBackedEmbeddingError.unsupportedDType(name)
    }
}

private func readSafetensorEntries(url: URL) throws -> [String: SafetensorEntry] {
    let data = try Data(contentsOf: url, options: .alwaysMapped)
    guard data.count >= 8 else {
        throw DiskBackedEmbeddingError.malformedSafetensors(url)
    }
    let headerLength = data.withUnsafeBytes {
        $0.loadUnaligned(as: UInt64.self).littleEndian
    }
    guard headerLength > 0, headerLength <= UInt64(data.count - 8) else {
        throw DiskBackedEmbeddingError.malformedSafetensors(url)
    }
    let headerEnd = 8 + Int(headerLength)
    guard
        let object = try JSONSerialization.jsonObject(with: data[8 ..< headerEnd])
            as? [String: Any]
    else {
        throw DiskBackedEmbeddingError.malformedSafetensors(url)
    }

    var result = [String: SafetensorEntry]()
    for (name, rawEntry) in object where name != "__metadata__" {
        guard let entry = rawEntry as? [String: Any],
            let dtypeName = entry["dtype"] as? String,
            let shapeNumbers = entry["shape"] as? [NSNumber],
            let offsets = entry["data_offsets"] as? [NSNumber], offsets.count == 2
        else {
            throw DiskBackedEmbeddingError.malformedSafetensors(url)
        }
        let shape = shapeNumbers.map(\.intValue)
        guard shape.count == 2, shape[0] > 0, shape[1] > 0 else {
            throw DiskBackedEmbeddingError.invalidShape(name, shape)
        }
        let begin = offsets[0].intValue
        let end = offsets[1].intValue
        guard begin >= 0, end >= begin, headerEnd + end <= data.count else {
            throw DiskBackedEmbeddingError.malformedSafetensors(url)
        }
        let byteCount = end - begin
        guard byteCount > 0, byteCount % shape[0] == 0 else {
            throw DiskBackedEmbeddingError.invalidShape(name, shape)
        }
        result[name] = try SafetensorEntry(
            name: name,
            dtypeName: dtypeName,
            dtype: safetensorDType(dtypeName),
            shape: shape,
            data: data,
            dataStart: headerEnd + begin,
            byteCount: byteCount)
    }
    return result
}

/// An embedding whose checkpoint table stays memory-mapped and whose requested rows alone are
/// copied into MLX unified memory.
///
/// This is intended for exceptionally wide vocabulary tables such as Gemma 4 PLE. It deliberately
/// owns a `[1, 1]` placeholder `weight` so normal strict checkpoint verification and model
/// materialization remain useful without realizing the real table.
package final class DiskBackedEmbedding: Embedding {
    package let storageTensorNames: Set<String>
    package let checkpointWeightName: String
    package let dimensions: Int
    package let vocabularySize: Int

    private let storedWeight: SafetensorEntry
    private let storedScales: SafetensorEntry?
    private let storedBiases: SafetensorEntry?
    private let bits: Int?
    private let groupSize: Int?
    private let quantizationMode: QuantizationMode

    package init(
        modelDirectory: URL,
        weightSuffix: String,
        dimensions: Int,
        vocabularySize: Int
    ) throws {
        var entries = [String: SafetensorEntry]()
        for url in try safetensorWeightURLs(in: modelDirectory) {
            entries.merge(try readSafetensorEntries(url: url)) { _, new in new }
        }

        let candidates = entries.keys.filter { $0.hasSuffix(weightSuffix) }.sorted()
        guard let weightName = candidates.first else {
            throw DiskBackedEmbeddingError.missingTensor(weightSuffix)
        }
        guard candidates.count == 1 else {
            throw DiskBackedEmbeddingError.ambiguousTensor(weightSuffix, candidates)
        }
        guard let weight = entries[weightName] else {
            throw DiskBackedEmbeddingError.missingTensor(weightSuffix)
        }
        let base = String(weightName.dropLast(".weight".count))
        let scalesName = "\(base).scales"
        let biasesName = "\(base).biases"
        let scales = entries[scalesName]
        let biases = entries[biasesName]

        guard weight.rowCount == vocabularySize else {
            throw DiskBackedEmbeddingError.invalidShape(weight.name, weight.shape)
        }

        let inferredBits: Int?
        let inferredGroupSize: Int?
        let mode: QuantizationMode
        if let scales {
            guard weight.dtypeName == "U32", scales.rowCount == vocabularySize,
                dimensions % scales.rowWidth == 0,
                weight.rowWidth * 32 % dimensions == 0
            else {
                throw DiskBackedEmbeddingError.incompatibleQuantization(
                    "Gemma PLE quantization geometry does not match the model configuration.")
            }
            if let biases, biases.shape != scales.shape {
                throw DiskBackedEmbeddingError.incompatibleQuantization(
                    "Gemma PLE bias and scale shapes differ.")
            }
            inferredBits = weight.rowWidth * 32 / dimensions
            inferredGroupSize = dimensions / scales.rowWidth
            if biases != nil {
                mode = .affine
            } else if inferredBits == 4, inferredGroupSize == 32 {
                mode = .mxfp4
            } else if inferredBits == 8, inferredGroupSize == 32 {
                mode = .mxfp8
            } else if inferredBits == 4, inferredGroupSize == 16 {
                mode = .nvfp4
            } else {
                throw DiskBackedEmbeddingError.incompatibleQuantization(
                    "Unsupported Gemma PLE quantization: bits=\(inferredBits!), groupSize=\(inferredGroupSize!)."
                )
            }
        } else {
            guard biases == nil, weight.rowWidth == dimensions,
                ["BF16", "F16", "F32"].contains(weight.dtypeName)
            else {
                throw DiskBackedEmbeddingError.incompatibleQuantization(
                    "Dense Gemma PLE geometry does not match the model configuration.")
            }
            inferredBits = nil
            inferredGroupSize = nil
            mode = .affine
        }

        self.storageTensorNames = Set(
            [weightName, scales == nil ? nil : scalesName, biases == nil ? nil : biasesName]
                .compactMap { $0 })
        self.checkpointWeightName = weightName
        self.dimensions = dimensions
        self.vocabularySize = vocabularySize
        self.storedWeight = weight
        self.storedScales = scales
        self.storedBiases = biases
        self.bits = inferredBits
        self.groupSize = inferredGroupSize
        self.quantizationMode = mode
        super.init(weight: MLXArray.zeros([1, 1]))
    }

    override package var shape: (Int, Int) {
        (vocabularySize, dimensions)
    }

    override package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let originalShape = x.shape
        let tokens = x.reshaped(-1).asArray(Int.self)

        var uniqueTokens = [Int]()
        var tokenToIndex = [Int: Int]()
        var inverse = [Int]()
        uniqueTokens.reserveCapacity(tokens.count)
        inverse.reserveCapacity(tokens.count)
        for token in tokens {
            if let index = tokenToIndex[token] {
                inverse.append(index)
            } else {
                let index = uniqueTokens.count
                tokenToIndex[token] = index
                uniqueTokens.append(token)
                inverse.append(index)
            }
        }

        do {
            var values = try storedWeight.rows(uniqueTokens)
            if let storedScales, let bits, let groupSize {
                let scales = try storedScales.rows(uniqueTokens)
                let biases = try storedBiases?.rows(uniqueTokens)
                values = dequantized(
                    values, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, mode: quantizationMode)
            }
            if uniqueTokens.count != tokens.count {
                values = values[MLXArray(inverse)]
            }
            return values.reshaped(originalShape + [dimensions])
        } catch {
            preconditionFailure("Disk-backed embedding read failed: \(error.localizedDescription)")
        }
    }
}
