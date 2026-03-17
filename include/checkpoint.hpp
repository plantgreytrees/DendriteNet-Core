#pragma once
#include "tensor.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstring>
#include <cstdint>

namespace dendrite {

// ============================================================
// Binary checkpoint format "DNRT0001"
//
// Layout (all integers little-endian):
//   [8  bytes]  magic "DNRT0001"
//   [4  bytes]  version = 1  (uint32)
//   [4  bytes]  num_entries  (uint32)
//   [8  bytes]  data_section_offset (uint64) — absolute byte position
//   Per entry (sequential):
//     [4  bytes]  name_len  (uint32)
//     [name_len]  name (UTF-8, no null terminator)
//     [4  bytes]  ndim  (uint32)
//     [8*ndim]    shape  (uint64[ndim])
//     [8  bytes]  byte_offset within data section  (uint64)
//     [8  bytes]  count = number of float32 values  (uint64)
//   Data section at data_section_offset:
//     Concatenated float32 arrays, native byte order (LE on x86)
// ============================================================

/// Collects named tensors and scalars, then writes them to a binary file.
class CheckpointWriter {
public:
    struct Entry {
        std::string name;
        std::vector<size_t> shape;
        std::vector<float>  data;
    };

    /// Register a tensor (value is copied at registration time).
    void add(const std::string& name, const Tensor& t) {
        entries_.push_back({name, t.shape, t.data});
    }

    /// Register a single float scalar (stored as a 1-element tensor).
    void add_scalar(const std::string& name, float v) {
        entries_.push_back({name, {1}, {v}});
    }

    /// Write all registered entries to path. Returns true on success.
    [[nodiscard]] bool save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) return false;

        // Compute per-entry byte offsets within the data section
        std::vector<uint64_t> byte_offsets(entries_.size());
        uint64_t data_running = 0;
        for (size_t i = 0; i < entries_.size(); i++) {
            byte_offsets[i] = data_running;
            data_running += static_cast<uint64_t>(entries_[i].data.size()) * sizeof(float);
        }

        // Pre-compute header size so we can write data_section_offset up front
        // Header = 8 + 4 + 4 + 8 = 24 bytes fixed
        // + per entry: 4 + name_len + 4 + 8*ndim + 8 + 8
        uint64_t hdr_size = 24;
        for (auto& e : entries_)
            hdr_size += 4 + static_cast<uint64_t>(e.name.size())
                      + 4 + 8 * static_cast<uint64_t>(e.shape.size())
                      + 8 + 8;

        // Write magic
        f.write("DNRT0001", 8);
        // Version
        const uint32_t ver = 1;
        f.write(reinterpret_cast<const char*>(&ver), 4);
        // num_entries
        const uint32_t n = static_cast<uint32_t>(entries_.size());
        f.write(reinterpret_cast<const char*>(&n), 4);
        // data_section_offset = hdr_size (header starts at byte 0)
        f.write(reinterpret_cast<const char*>(&hdr_size), 8);

        // Entry headers
        for (size_t i = 0; i < entries_.size(); i++) {
            const auto& e = entries_[i];
            const uint32_t name_len = static_cast<uint32_t>(e.name.size());
            f.write(reinterpret_cast<const char*>(&name_len), 4);
            f.write(e.name.data(), name_len);
            const uint32_t ndim = static_cast<uint32_t>(e.shape.size());
            f.write(reinterpret_cast<const char*>(&ndim), 4);
            for (size_t d : e.shape) {
                const uint64_t dv = static_cast<uint64_t>(d);
                f.write(reinterpret_cast<const char*>(&dv), 8);
            }
            f.write(reinterpret_cast<const char*>(&byte_offsets[i]), 8);
            const uint64_t cnt = static_cast<uint64_t>(e.data.size());
            f.write(reinterpret_cast<const char*>(&cnt), 8);
        }

        // Data section
        for (const auto& e : entries_)
            f.write(reinterpret_cast<const char*>(e.data.data()),
                    static_cast<std::streamsize>(e.data.size() * sizeof(float)));

        return f.good();
    }

    size_t num_entries() const { return entries_.size(); }

private:
    std::vector<Entry> entries_;
};

/// Loads a checkpoint file and restores tensors by name.
class CheckpointReader {
public:
    struct Entry {
        std::string        name;
        std::vector<size_t> shape;
        std::vector<float>  data;
    };

    /// Load checkpoint at path. Returns true on success.
    [[nodiscard]] bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;

        // Magic
        char magic[8];
        f.read(magic, 8);
        if (!f.good() || std::strncmp(magic, "DNRT0001", 8) != 0) return false;

        // Version
        uint32_t ver = 0;
        f.read(reinterpret_cast<char*>(&ver), 4);
        if (ver != 1) return false;

        // num_entries
        uint32_t n = 0;
        f.read(reinterpret_cast<char*>(&n), 4);

        // data_section_offset
        uint64_t data_off = 0;
        f.read(reinterpret_cast<char*>(&data_off), 8);

        // Read entry headers
        struct RawEntry {
            std::string        name;
            std::vector<size_t> shape;
            uint64_t           byte_offset;
            uint64_t           count;
        };
        std::vector<RawEntry> raw(n);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t name_len = 0;
            f.read(reinterpret_cast<char*>(&name_len), 4);
            raw[i].name.resize(name_len);
            f.read(raw[i].name.data(), name_len);
            uint32_t ndim = 0;
            f.read(reinterpret_cast<char*>(&ndim), 4);
            raw[i].shape.resize(ndim);
            for (uint32_t d = 0; d < ndim; d++) {
                uint64_t dv = 0;
                f.read(reinterpret_cast<char*>(&dv), 8);
                raw[i].shape[d] = static_cast<size_t>(dv);
            }
            f.read(reinterpret_cast<char*>(&raw[i].byte_offset), 8);
            f.read(reinterpret_cast<char*>(&raw[i].count), 8);
            if (!f.good()) return false;
        }

        // Read data for each entry
        entries_.clear();
        index_.clear();
        entries_.reserve(n);
        for (const auto& re : raw) {
            f.seekg(static_cast<std::streamoff>(data_off + re.byte_offset));
            std::vector<float> data(re.count);
            f.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(re.count * sizeof(float)));
            if (!f.good()) return false;
            const size_t idx = entries_.size();
            entries_.push_back({re.name, re.shape, std::move(data)});
            index_[re.name] = idx;
        }
        return true;
    }

    bool has(const std::string& name) const { return index_.count(name) > 0; }

    /// Restore tensor t from the entry named 'name'.
    /// Resizes t if shape differs. Applies NaN guard on load. Returns false if not found.
    bool restore(const std::string& name, Tensor& t) const {
        const auto it = index_.find(name);
        if (it == index_.end()) return false;
        const auto& e = entries_[it->second];
        if (t.shape != e.shape)
            t = Tensor(e.shape);
        t.data = e.data;
        for (auto& v : t.data) if (!std::isfinite(v)) v = 0.0f;
        return true;
    }

    /// Restore a scalar saved with add_scalar(). Returns false if not found.
    bool restore_scalar(const std::string& name, float& v) const {
        const auto it = index_.find(name);
        if (it == index_.end()) return false;
        const auto& e = entries_[it->second];
        if (e.data.empty()) return false;
        v = e.data[0];
        return true;
    }

    size_t num_entries() const { return entries_.size(); }

private:
    std::vector<Entry>                  entries_;
    std::unordered_map<std::string, size_t> index_;
};

} // namespace dendrite
