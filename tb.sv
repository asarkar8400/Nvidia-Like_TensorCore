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
        .clk         (clk),
        .rst_n       (rst_n),
        .mma_enable  (mma_enable),
        .mma_valid   (mma_valid),
        .mma_ready   (mma_ready),
        .result_valid(result_valid),
        .matrix_a    (A),
        .matrix_b    (B),
        .matrix_c    (C),
        .matrix_d    (D)
    );

    // Clock
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Helper: print FP16 as approximate real using same converter
    function automatic shortreal fp16_to_shortreal (input logic [15:0] h);
        int  s;
        int  e;
        int  f;
        real r;
        const int BIAS = 15;

        s = h[15];
        e = h[14:10];
        f = h[9:0];

        if (e == 0) begin
            if (f == 0)
                r = 0.0;
            else
                r = (s ? -1.0 : 1.0) * (2.0 ** (1 - BIAS)) * (f / 1024.0);
        end
        else if (e == 31) begin
            r = (s ? -1.0 : 1.0) * 1.0e30;
        end
        else begin
            r = (s ? -1.0 : 1.0) *
                (2.0 ** (e - BIAS)) *
                (1.0 + f / 1024.0);
        end
        fp16_to_shortreal = shortreal'(r);
    endfunction

    // --- Print helpers (unchanged from your version) ---

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
                $write(" %6.3f", fp16_to_shortreal(m[idx]));
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
                $write(" %8.4f", val);
            end
            $display(" ]");
        end
    endtask

    // ========================================================
    // Stimulus with more interesting FP values
    // ========================================================
    initial begin
        int i, j, idx;

        $display("=== FP16 Tensor Core Tile Testbench ===");

        mma_enable = 1'b1;
        mma_valid  = 1'b0;
        rst_n      = 1'b0;

        // ------------------------------------
        // A: 4x4 FP16 with varied values
        // row 0: [5.823, -1.75, 0.333, 2.718]
        // row 1: [4.212, 3.1416, -0.5, 1.125]
        // row 2: [0.125, -2.25, 7.75, -3.5]
        // row 3: [10.5, -0.875, 1.75, -4.0]
        // (FP16 hex precomputed)
        // ------------------------------------
        A[ 0] = 16'h45D3; //  5.823
        A[ 1] = 16'hBF00; // -1.75
        A[ 2] = 16'h3554; //  0.333
        A[ 3] = 16'h4170; //  2.718

        A[ 4] = 16'h4436; //  4.212
        A[ 5] = 16'h4248; //  3.1416
        A[ 6] = 16'hB800; // -0.5
        A[ 7] = 16'h3C80; //  1.125

        A[ 8] = 16'h3000; //  0.125
        A[ 9] = 16'hC080; // -2.25
        A[10] = 16'h47C0; //  7.75
        A[11] = 16'hC300; // -3.5

        A[12] = 16'h4940; // 10.5
        A[13] = 16'hBB00; // -0.875
        A[14] = 16'h3F00; //  1.75
        A[15] = 16'hC400; // -4.0

        // ------------------------------------
        // B: 4x4 FP16 with different varied values
        // row 0: [0.5, -1.25, 3.5, -0.75]
        // row 1: [2.25, -3.125, 4.5, 1.375]
        // row 2: [-2.0, 5.25, -1.875, 0.625]
        // row 3: [6.5, -4.25, 0.75, -5.5]
        // ------------------------------------
        B[ 0] = 16'h3800; //  0.5
        B[ 1] = 16'hBD00; // -1.25
        B[ 2] = 16'h4300; //  3.5
        B[ 3] = 16'hBA00; // -0.75

        B[ 4] = 16'h4080; //  2.25
        B[ 5] = 16'hC240; // -3.125
        B[ 6] = 16'h4480; //  4.5
        B[ 7] = 16'h3D80; //  1.375

        B[ 8] = 16'hC000; // -2.0
        B[ 9] = 16'h4540; //  5.25
        B[10] = 16'hBF80; // -1.875
        B[11] = 16'h3900; //  0.625

        B[12] = 16'h4680; //  6.5
        B[13] = 16'hC440; // -4.25
        B[14] = 16'h3A00; //  0.75
        B[15] = 16'hC580; // -5.5

        // ------------------------------------
        // C: 4x4 FP32 (via $shortrealtobits) with varied values
        // row 0: [ 1.25, -0.75,  0.0,  2.5  ]
        // row 1: [-1.00,  3.75,  0.5, -2.25 ]
        // row 2: [ 4.125,-3.50,  1.0,  0.875]
        // row 3: [ 0.25, -1.50,  2.0, -0.5  ]
        // ------------------------------------
        C[ 0] = $shortrealtobits(1.25);
        C[ 1] = $shortrealtobits(-0.75);
        C[ 2] = $shortrealtobits(0.0);
        C[ 3] = $shortrealtobits(2.5);

        C[ 4] = $shortrealtobits(-1.0);
        C[ 5] = $shortrealtobits(3.75);
        C[ 6] = $shortrealtobits(0.5);
        C[ 7] = $shortrealtobits(-2.25);

        C[ 8] = $shortrealtobits(4.125);
        C[ 9] = $shortrealtobits(-3.5);
        C[10] = $shortrealtobits(1.0);
        C[11] = $shortrealtobits(0.875);

        C[12] = $shortrealtobits(0.25);
        C[13] = $shortrealtobits(-1.5);
        C[14] = $shortrealtobits(2.0);
        C[15] = $shortrealtobits(-0.5);

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

        $display("\n=== Test Complete ===");
        #50;
        $finish;
    end

    // Timeout
    initial begin
        #100000;
        $display("ERROR: Simulation timeout.");
        $finish;
    end

endmodule
