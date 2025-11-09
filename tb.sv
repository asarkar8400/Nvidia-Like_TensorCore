`timescale 1ns/1ps

//============================================================
// Testbench for tensor_core_fp16_mma
//============================================================

module tb_tensor_core_fp16_mma;

    localparam int M          = 4;
    localparam int N          = 4;
    localparam int K          = 4;
    localparam int A_ELEMS    = M*K;
    localparam int B_ELEMS    = K*N;
    localparam int C_ELEMS    = M*N;
    localparam int CLK_PERIOD = 10;

    // Clock / reset
    logic clk;
    logic rst_n;

    // Control
    logic mma_enable;
    logic mma_valid;
    logic mma_ready;
    logic result_valid;

    // Matrices
    logic [15:0]  A [0:A_ELEMS-1]; // FP16
    logic [15:0]  B [0:B_ELEMS-1]; // FP16
    logic [31:0]  C [0:C_ELEMS-1]; // FP32
    logic [31:0]  D [0:C_ELEMS-1]; // FP32 output

    // DUT
    tensor_core_fp16_mma #(
        .M (M),
        .N (N),
        .K (K)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .mma_enable (mma_enable),
        .mma_valid  (mma_valid),
        .mma_ready  (mma_ready),
        .result_valid (result_valid),
        .matrix_a   (A),
        .matrix_b   (B),
        .matrix_c   (C),
        .matrix_d   (D)
    );

    // Clock
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // FP16 constants for small integers
    localparam logic [15:0] FP16_1_0 = 16'h3C00; // 1.0
    localparam logic [15:0] FP16_2_0 = 16'h4000; // 2.0

    // Helper: print FP16 as approximate real using same converter
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
            if (f == 0) r = 0.0;
            else r = (s ? -1.0 : 1.0) * (2.0 ** (1 - BIAS)) * (f / 1024.0);
        end
        else if (e == 31) begin
            r = (s ? -1.0 : 1.0) * 1.0e30;
        end
        else begin
            r = (s ? -1.0 : 1.0) * (2.0 ** (e - BIAS)) * (1.0 + f / 1024.0);
        end
        fp16_to_shortreal = shortreal'(r);
    endfunction

    task automatic print_matrix_fp16(
        input string name,
        input logic [15:0] m[],
        input int rows,
        input int cols
    );
        int i, j, idx;
        $display("%s (FP16 values, printed as float):", name);
        for (i = 0; i < rows; i++) begin
            $write("  [");
            for (j = 0; j < cols; j++) begin
                idx = i*cols + j;
                $write(" %6.2f", fp16_to_shortreal(m[idx]));
            end
            $display(" ]");
        end
    endtask

    task automatic print_matrix_fp32_bits_as_real(
        input string name,
        input logic [31:0] m[],
        input int rows,
        input int cols
    );
        int i, j, idx;
        shortreal val;
        $display("%s (FP32):", name);
        for (i = 0; i < rows; i++) begin
            $write("  [");
            for (j = 0; j < cols; j++) begin
                idx = i*cols + j;
                val = $bitstoshortreal(m[idx]);
                $write(" %7.3f", val);
            end
            $display(" ]");
        end
    endtask

    // Stimulus
    initial begin
        int i, j, idx;

        $display("=== FP16 Tensor Core Tile Testbench ===");

        mma_enable = 1'b1;
        mma_valid  = 1'b0;
        rst_n      = 1'b0;

        // A: near-identity (2.0 on diagonal, 1.0 elsewhere)
        for (i = 0; i < M; i++) begin
            for (j = 0; j < K; j++) begin
                idx = i*K + j;
                if (i == j) A[idx] = FP16_2_0; // 2.0
                else        A[idx] = FP16_1_0; // 1.0
            end
        end

        // B: all ones
        for (i = 0; i < K; i++) begin
            for (j = 0; j < N; j++) begin
                idx = i*N + j;
                B[idx] = FP16_1_0; // 1.0
            end
        end

        // C (FP32): mostly zeros, with a few offsets: 3, 2, 5
        for (idx = 0; idx < C_ELEMS; idx++) begin
            C[idx] = $shortrealtobits(0.0);
        end
        C[0*N + 0] = $shortrealtobits(3.0); // (0,0)
        C[2*N + 0] = $shortrealtobits(2.0); // (2,0)
        C[2*N + 2] = $shortrealtobits(5.0); // (2,2)

        // Reset
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // Print inputs
        $display("");
        print_matrix_fp16("A", A, M, K);
        $display("");
        print_matrix_fp16("B", B, K, N);

        $display("");
        print_matrix_fp32_bits_as_real("C", C, M, N);

        // Start MMA
        $display("\nStarting FP16 MMA: D = A * B + C ...");
        @(posedge clk);
        mma_valid = 1'b1;
        @(posedge clk);
        mma_valid = 1'b0;

        // Wait for result_valid
        @(posedge result_valid);
        @(posedge clk); // extra cycle

        $display("\nResult matrix D:");
        print_matrix_fp32_bits_as_real("D", D, M, N);

        $display("\nExpected behavior (approximately):");
        $display("  A row-sum ≈ 5.0, B all 1.0    => A*B ≈ all 5.0");
        $display("  C adds offsets: (0,0)+3 => 8.0, (2,0)+2 => 7.0, (2,2)+5 => 10.0");

        $display("\n=== Test Complete ===");
        #50;
        $finish;
    end

    // Timeout
    initial begin
        #100000;
        $display(\"ERROR: Simulation timeout.\");
        $finish;
    end

endmodule
