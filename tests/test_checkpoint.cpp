#include "../include/checkpoint.hpp"
#include "../include/layer.hpp"
#include "../include/dendrite3d.hpp"
#include "test_runner.hpp"
#include <cstdio>
#include <cmath>

using namespace dendrite;

// ---- CheckpointWriter / CheckpointReader --------------------------------

TEST(checkpoint_write_read_tensor) {
    Tensor t({3, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    CheckpointWriter wr;
    wr.add("my_tensor", t);
    ASSERT(wr.save("./ckpt_test_ckpt.dnrt"));

    CheckpointReader rd;
    ASSERT(rd.load("./ckpt_test_ckpt.dnrt"));
    ASSERT(rd.has("my_tensor"));

    Tensor out({3, 2});
    ASSERT(rd.restore("my_tensor", out));
    ASSERT_EQ(out.shape[0], 3u);
    ASSERT_EQ(out.shape[1], 2u);
    for (size_t i = 0; i < 6; i++) ASSERT_NEAR(out[i], t[i], 1e-6f);
}

TEST(checkpoint_write_read_scalar) {
    CheckpointWriter wr;
    wr.add_scalar("step", 42.f);
    wr.add_scalar("lr",   0.003f);
    ASSERT(wr.save("./ckpt_test_scalar.dnrt"));

    CheckpointReader rd;
    ASSERT(rd.load("./ckpt_test_scalar.dnrt"));

    float step = 0.f, lr = 0.f;
    ASSERT(rd.restore_scalar("step", step));
    ASSERT(rd.restore_scalar("lr",   lr));
    ASSERT_NEAR(step, 42.f,   1e-6f);
    ASSERT_NEAR(lr,   0.003f, 1e-6f);
}

TEST(checkpoint_missing_key_returns_false) {
    CheckpointWriter wr;
    wr.add_scalar("x", 1.f);
    (void)wr.save("./ckpt_test_miss.dnrt");

    CheckpointReader rd;
    (void)rd.load("./ckpt_test_miss.dnrt");
    ASSERT(!rd.has("y"));
    float v = 99.f;
    ASSERT(!rd.restore_scalar("y", v));
    ASSERT_NEAR(v, 99.f, 1e-6f);  // unchanged
}

TEST(checkpoint_bad_magic_rejected) {
    // Write a file with wrong magic
    {
        std::ofstream f("./ckpt_test_bad.dnrt", std::ios::binary);
        f.write("BADSIG!!", 8);
        uint32_t z = 0; f.write(reinterpret_cast<const char*>(&z), 4);
    }
    CheckpointReader rd;
    ASSERT(!rd.load("./ckpt_test_bad.dnrt"));
}

TEST(checkpoint_multiple_tensors_offsets) {
    CheckpointWriter wr;
    Tensor a({4}, {1.f, 2.f, 3.f, 4.f});
    Tensor b({2, 3}, {5.f, 6.f, 7.f, 8.f, 9.f, 10.f});
    wr.add("a", a);
    wr.add("b", b);
    ASSERT(wr.save("./ckpt_test_multi.dnrt"));

    CheckpointReader rd;
    ASSERT(rd.load("./ckpt_test_multi.dnrt"));
    Tensor ra({4}), rb({2, 3});
    ASSERT(rd.restore("a", ra));
    ASSERT(rd.restore("b", rb));
    for (size_t i = 0; i < 4; i++) ASSERT_NEAR(ra[i], a[i], 1e-6f);
    for (size_t i = 0; i < 6; i++) ASSERT_NEAR(rb[i], b[i], 1e-6f);
}

TEST(checkpoint_nan_guard_on_load) {
    // Manually write a file with NaN values
    CheckpointWriter wr;
    Tensor t({3}, {1.f, std::numeric_limits<float>::quiet_NaN(), 3.f});
    wr.add("nan_tensor", t);
    (void)wr.save("./ckpt_test_nan.dnrt");

    CheckpointReader rd;
    (void)rd.load("./ckpt_test_nan.dnrt");
    Tensor out({3});
    rd.restore("nan_tensor", out);
    // NaN guard should replace NaN with 0
    ASSERT_FINITE(out[0]);
    ASSERT_FINITE(out[1]);
    ASSERT_NEAR(out[1], 0.f, 1e-6f);
    ASSERT_FINITE(out[2]);
}

// ---- DenseLayer serialize / deserialize --------------------------------

TEST(dense_layer_serialize_round_trip) {
    std::mt19937 rng(42);
    DenseLayer layer(4, 3, Activation::RELU, rng);
    // Give it some non-trivial Adam state
    layer.adam_t = 7;
    layer.m_w[0] = 0.5f; layer.v_w[1] = 0.25f;

    CheckpointWriter wr;
    layer.serialize(wr, "lyr_");
    (void)wr.save("./ckpt_test_dense.dnrt");

    // Create a fresh layer with same dims and restore
    DenseLayer loaded(4, 3, Activation::RELU, rng);
    loaded.adam_t = 0;
    CheckpointReader rd;
    (void)rd.load("./ckpt_test_dense.dnrt");
    loaded.deserialize(rd, "lyr_");

    ASSERT_EQ(loaded.adam_t, 7);
    for (size_t i = 0; i < layer.weights.size(); i++)
        ASSERT_NEAR(loaded.weights[i], layer.weights[i], 1e-6f);
    ASSERT_NEAR(loaded.m_w[0], 0.5f,  1e-6f);
    ASSERT_NEAR(loaded.v_w[1], 0.25f, 1e-6f);
}

// ---- MiniNetwork serialize / deserialize --------------------------------

TEST(mini_network_serialize_round_trip) {
    std::mt19937 rng(99);
    MiniNetwork net("test", {8, 16, 4}, Activation::RELU, Activation::NONE, rng);

    CheckpointWriter wr;
    net.serialize(wr, "mn_");
    (void)wr.save("./ckpt_test_mini.dnrt");

    MiniNetwork loaded("test", {8, 16, 4}, Activation::RELU, Activation::NONE, rng);
    CheckpointReader rd;
    (void)rd.load("./ckpt_test_mini.dnrt");
    loaded.deserialize(rd, "mn_");

    // Verify weights match for all layers
    for (size_t l = 0; l < net.layers.size(); l++) {
        for (size_t i = 0; i < net.layers[l].weights.size(); i++)
            ASSERT_NEAR(loaded.layers[l].weights[i], net.layers[l].weights[i], 1e-6f);
    }
}

// ---- Tensor::matmul -------------------------------------------------------

TEST(matmul_small_correct) {
    // A [2×3] × B [3×2] = C [2×2]
    Tensor A({2, 3}, {1.f, 2.f, 3.f,
                      4.f, 5.f, 6.f});
    Tensor B({3, 2}, {7.f,  8.f,
                      9.f,  10.f,
                      11.f, 12.f});
    Tensor C = Tensor::matmul(A, B);
    ASSERT_EQ(C.shape[0], 2u);
    ASSERT_EQ(C.shape[1], 2u);
    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    ASSERT_NEAR(C.at(0,0), 58.f,  1e-4f);
    ASSERT_NEAR(C.at(0,1), 64.f,  1e-4f);
    // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    ASSERT_NEAR(C.at(1,0), 139.f, 1e-4f);
    ASSERT_NEAR(C.at(1,1), 154.f, 1e-4f);
}

TEST(matmul_identity) {
    // A × I = A
    Tensor A({3, 3}, {1.f,2.f,3.f, 4.f,5.f,6.f, 7.f,8.f,9.f});
    Tensor I({3, 3}); I.at(0,0)=1.f; I.at(1,1)=1.f; I.at(2,2)=1.f;
    Tensor C = Tensor::matmul(A, I);
    for (size_t i = 0; i < 9; i++) ASSERT_NEAR(C[i], A[i], 1e-5f);
}

// ---- DenseLayer forward_batch / backward_batch -------------------------

TEST(forward_batch_matches_single) {
    std::mt19937 rng(7);
    DenseLayer layer(4, 3, Activation::RELU, rng);
    const size_t B = 5;
    std::vector<Tensor> inputs(B);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t b = 0; b < B; b++) {
        inputs[b] = Tensor({4});
        for (size_t k = 0; k < 4; k++) inputs[b][k] = nd(rng);
    }

    // Single-sample forward for each
    std::vector<Tensor> single_outs(B);
    for (size_t b = 0; b < B; b++) single_outs[b] = layer.forward(inputs[b]);

    // Batched forward
    Tensor in_batch({B, 4});
    for (size_t b = 0; b < B; b++)
        for (size_t k = 0; k < 4; k++) in_batch.data[b * 4 + k] = inputs[b][k];
    Tensor out_batch = layer.forward_batch(in_batch);

    for (size_t b = 0; b < B; b++)
        for (size_t j = 0; j < 3; j++)
            ASSERT_NEAR(out_batch.data[b * 3 + j], single_outs[b][j], 1e-5f);
}

TEST(backward_batch_gradient_not_nan) {
    std::mt19937 rng(13);
    DenseLayer layer(6, 4, Activation::RELU, rng);
    const size_t B = 8;
    Tensor in_b({B, 6}); in_b.he_init(rng);
    layer.forward_batch(in_b);

    Tensor grad_b({B, 4});
    std::normal_distribution<float> nd(0.f, 0.1f);
    for (auto& v : grad_b.data) v = nd(rng);

    Tensor grad_in = layer.backward_batch(grad_b);
    ASSERT_EQ(grad_in.shape[0], B);
    ASSERT_EQ(grad_in.shape[1], 6u);
    for (auto& v : grad_in.data) ASSERT_FINITE(v);
    for (auto& v : layer.grad_w.data) ASSERT_FINITE(v);
    for (auto& v : layer.grad_b.data) ASSERT_FINITE(v);
}

// ---- DendriteNet3D save_checkpoint / load_checkpoint -------------------

TEST(dendritenet_checkpoint_round_trip) {
    std::mt19937 rng(42);
    DendriteNet3D net(4, 3, 42);
    net.build({"a", "b"}, 16);

    // Train a few steps so weights are non-trivial
    std::vector<Tensor> xs, ys;
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < 10; i++) {
        Tensor x({4}); for (auto& v : x.data) v = nd(rng);
        Tensor y({3}); y[i % 3] = 1.f;
        xs.push_back(x); ys.push_back(y);
    }
    net.train_batch(xs, ys, 2);

    // Save
    ASSERT(net.save_checkpoint("./ckpt_test_net.dnrt"));

    // Capture weights before loading
    float w0_before = net.branches[0]->specialist.layers[0].weights[0];
    size_t steps_before = net.total_train_steps;

    // Corrupt the first weight to verify restore works
    net.branches[0]->specialist.layers[0].weights[0] = 999.f;

    // Load
    ASSERT(net.load_checkpoint("./ckpt_test_net.dnrt"));

    // Verify weight was restored
    ASSERT_NEAR(net.branches[0]->specialist.layers[0].weights[0], w0_before, 1e-6f);
    ASSERT_EQ(net.total_train_steps, steps_before);
}

// ---- DendriteNet3D::train_minibatch ------------------------------------

TEST(train_minibatch_returns_finite_loss) {
    DendriteNet3D net(4, 3, 1);
    net.build({"a", "b"}, 16);

    std::mt19937 rng(55);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<Tensor> xs, ys;
    for (int i = 0; i < 32; i++) {
        Tensor x({4}); for (auto& v : x.data) v = nd(rng);
        Tensor y({3}); y[i % 3] = 1.f;
        xs.push_back(x); ys.push_back(y);
    }

    float loss = net.train_minibatch(xs, ys);
    ASSERT_FINITE(loss);
    ASSERT_GT(loss, 0.f);
}

TEST(train_minibatch_reduces_loss) {
    DendriteNet3D net(4, 3, 2);
    net.learning_rate = 0.01f;
    net.build({"a", "b"}, 32);

    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<Tensor> xs, ys;
    for (int i = 0; i < 64; i++) {
        Tensor x({4}); for (auto& v : x.data) v = nd(rng);
        Tensor y({3}); y[i % 3] = 1.f;
        xs.push_back(x); ys.push_back(y);
    }

    float loss_first  = net.train_minibatch(xs, ys);
    float loss_second = 0.f;
    for (int i = 0; i < 20; i++) loss_second = net.train_minibatch(xs, ys);
    // Loss should decrease with repeated passes
    ASSERT_LT(loss_second, loss_first);
}

// ---- main ---------------------------------------------------------------

int main() {
    std::cout << "=== Checkpoint & Batched GEMM Tests ===\n";
    RUN_TEST(checkpoint_write_read_tensor);
    RUN_TEST(checkpoint_write_read_scalar);
    RUN_TEST(checkpoint_missing_key_returns_false);
    RUN_TEST(checkpoint_bad_magic_rejected);
    RUN_TEST(checkpoint_multiple_tensors_offsets);
    RUN_TEST(checkpoint_nan_guard_on_load);
    RUN_TEST(dense_layer_serialize_round_trip);
    RUN_TEST(mini_network_serialize_round_trip);
    RUN_TEST(matmul_small_correct);
    RUN_TEST(matmul_identity);
    RUN_TEST(forward_batch_matches_single);
    RUN_TEST(backward_batch_gradient_not_nan);
    RUN_TEST(dendritenet_checkpoint_round_trip);
    RUN_TEST(train_minibatch_returns_finite_loss);
    RUN_TEST(train_minibatch_reduces_loss);
    return report("checkpoint");
}
