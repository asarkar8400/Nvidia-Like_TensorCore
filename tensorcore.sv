//============================================================
// tensor_core_fp16_mma
// NVIDIA-like tensor core tile (behavioral):
//   - FP16 inputs (A,B)
//   - FP32 accumulator (C) and outputs (D)
//   - Computes D = A(MxK) * B(KxN) + C(MxN)
//   - Multi-cycle over K dimension.
//============================================================

module tensor_core_fp16_mma #(
    parameter int M = 4,
    parameter int N = 4,
    parameter int K = 4,
    // element counts
    parameter int A_ELEMS = M*K,
    parameter int B_ELEMS = K*N,
    parameter int C_ELEMS = M*N
) (
    input  logic clk,
    input  logic rst_n,

    // Control
    input  logic mma_enable,
    input  logic mma_valid,
    output logic mma_ready,
    output logic result_valid,

    // FP16 input fragments (flattened)
    // A: M x K
    input  logic [15:0] matrix_a [0:A_ELEMS-1],
    // B: K x N
    input  logic [15:0] matrix_b [0:B_ELEMS-1],

    // FP32 accumulator
    // C: M x N
    input  logic [31:0] matrix_c [0:C_ELEMS-1],

    // FP32 result
    // D: M x N
    output logic [31:0] matrix_d [0:C_ELEMS-1]
);

    // Internal numeric representations
    shortreal a_sr  [0:A_ELEMS-1];  // A in FP32
    shortreal b_sr  [0:B_ELEMS-1];  // B in FP32
    shortreal acc   [0:C_ELEMS-1];  // accumulators (FP32)
    shortreal prod;                 // temp product

    // FSM
    typedef enum logic [1:0] {
        IDLE      = 2'b00,
        MAC       = 2'b01,
        WRITEBACK = 2'b10
    } state_t;

    state_t state;

    integer i, j, idx;
    integer k_idx;  // current K index (0..K-1)

    // --------------------------------------------------------
    // FP16 -> shortreal (approx IEEE-754 half-precision)
    //   sign (1), exponent (5), fraction (10), bias = 15
    //   This is not fully NaN/Inf-accurate but good enough
    //   for learning / basic compute.
    // --------------------------------------------------------
    function automatic shortreal fp16_to_shortreal (input logic [15:0] h);
        int s;
        int e;
        int f;
        real r;
        const int BIAS = 15;

        s = h[15];
        e = h[14:10];
        f = h[9:0];

        if (e == 0) begin
            // zero or subnormal
            if (f == 0) begin
                r = 0.0;
            end else begin
                // subnormal: (-1)^s * 2^(1-BIAS) * (f / 2^10)
                r = (s ? -1.0 : 1.0) *
                    (2.0 ** (1 - BIAS)) *
                    (f / 1024.0);
            end
        end
        else if (e == 31) begin
            // NaN or Inf: just saturate to large magnitude
            r = (s ? -1.0 : 1.0) * 1.0e30;
        end
        else begin
            // normal: (-1)^s * 2^(e-BIAS) * (1 + f/2^10)
            r = (s ? -1.0 : 1.0) *
                (2.0 ** (e - BIAS)) *
                (1.0 + f / 1024.0);
        end

        fp16_to_shortreal = shortreal'(r);
    endfunction

    // --------------------------------------------------------
    // FSM + arithmetic
    // --------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= IDLE;
            mma_ready    <= 1'b1;
            result_valid <= 1'b0;
            k_idx        <= 0;

            for (idx = 0; idx < C_ELEMS; idx = idx + 1) begin
                matrix_d[idx] <= 32'd0;
                acc[idx]      <= 0.0;
            end
        end
        else begin
            case (state)
                //------------------------------------------------
                // IDLE: wait for mma_valid && mma_enable,
                // then load/convert A,B,C into internal FP32.
                //------------------------------------------------
                IDLE: begin
                    mma_ready    <= 1'b1;
                    result_valid <= 1'b0;

                    if (mma_enable && mma_valid) begin
                        // Convert A, B FP16 -> FP32 (shortreal)
                        for (idx = 0; idx < A_ELEMS; idx = idx + 1) begin
                            a_sr[idx] <= fp16_to_shortreal(matrix_a[idx]);
                        end
                        for (idx = 0; idx < B_ELEMS; idx = idx + 1) begin
                            b_sr[idx] <= fp16_to_shortreal(matrix_b[idx]);
                        end
                        // Initialize accumulators from C (FP32 bits)
                        for (idx = 0; idx < C_ELEMS; idx = idx + 1) begin
                            acc[idx] <= $bitstoshortreal(matrix_c[idx]);
                        end

                        k_idx     <= 0;
                        mma_ready <= 1'b0;
                        state     <= MAC;
                    end
                end

                //------------------------------------------------
                // MAC: for each cycle, process one K-slice:
                //   acc[i,j] += A[i,k_idx] * B[k_idx,j]
                // After K cycles, go to WRITEBACK.
                //------------------------------------------------
                MAC: begin
                    integer row, col;
                    // One K step per cycle
                    for (row = 0; row < M; row = row + 1) begin
                        for (col = 0; col < N; col = col + 1) begin
                            int acc_idx;
                            int a_idx;
                            int b_idx;

                            acc_idx = row*N + col;
                            a_idx   = row*K + k_idx;   // (row, k_idx)
                            b_idx   = k_idx*N + col;   // (k_idx, col)

                            prod = a_sr[a_idx] * b_sr[b_idx];
                            acc[acc_idx] <= acc[acc_idx] + prod;
                        end
                    end

                    if (k_idx == K-1) begin
                        state <= WRITEBACK;
                    end
                    else begin
                        k_idx <= k_idx + 1;
                    end
                end

                //------------------------------------------------
                // WRITEBACK: convert accumulators to FP32 bits
                // and assert result_valid for one cycle.
                //------------------------------------------------
                WRITEBACK: begin
                    for (idx = 0; idx < C_ELEMS; idx = idx + 1) begin
                        matrix_d[idx] <= $shortrealtobits(acc[idx]);
                    end

                    result_valid <= 1'b1;
                    mma_ready    <= 1'b1;
                    state        <= IDLE;
                end

                default: begin
                    state        <= IDLE;
                    mma_ready    <= 1'b1;
                    result_valid <= 1'b0;
                end
            endcase
        end
    end

endmodule
