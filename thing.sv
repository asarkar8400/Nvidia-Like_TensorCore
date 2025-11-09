// NVIDIA Tensor Core Implementation (Icarus Verilog Compatible)
// Simplified INT8 version for compatibility

module tensor_core_int8 #(
    parameter MATRIX_DIM = 4,
    parameter INT8_WIDTH = 8,
    parameter INT32_WIDTH = 32
) (
    input  wire clk,
    input  wire rst_n,
    
    input  wire mma_enable,
    input  wire mma_valid,
    output reg mma_ready,
    output reg result_valid,
    
    // INT8 input matrices (flattened for compatibility)
    input  wire signed [INT8_WIDTH-1:0] matrix_a_00, matrix_a_01, matrix_a_02, matrix_a_03,
    input  wire signed [INT8_WIDTH-1:0] matrix_a_10, matrix_a_11, matrix_a_12, matrix_a_13,
    input  wire signed [INT8_WIDTH-1:0] matrix_a_20, matrix_a_21, matrix_a_22, matrix_a_23,
    input  wire signed [INT8_WIDTH-1:0] matrix_a_30, matrix_a_31, matrix_a_32, matrix_a_33,
    
    input  wire signed [INT8_WIDTH-1:0] matrix_b_00, matrix_b_01, matrix_b_02, matrix_b_03,
    input  wire signed [INT8_WIDTH-1:0] matrix_b_10, matrix_b_11, matrix_b_12, matrix_b_13,
    input  wire signed [INT8_WIDTH-1:0] matrix_b_20, matrix_b_21, matrix_b_22, matrix_b_23,
    input  wire signed [INT8_WIDTH-1:0] matrix_b_30, matrix_b_31, matrix_b_32, matrix_b_33,
    
    // INT32 accumulator and output (flattened)
    input  wire signed [INT32_WIDTH-1:0] matrix_c_00, matrix_c_01, matrix_c_02, matrix_c_03,
    input  wire signed [INT32_WIDTH-1:0] matrix_c_10, matrix_c_11, matrix_c_12, matrix_c_13,
    input  wire signed [INT32_WIDTH-1:0] matrix_c_20, matrix_c_21, matrix_c_22, matrix_c_23,
    input  wire signed [INT32_WIDTH-1:0] matrix_c_30, matrix_c_31, matrix_c_32, matrix_c_33,
    
    output reg signed [INT32_WIDTH-1:0] matrix_d_00, matrix_d_01, matrix_d_02, matrix_d_03,
    output reg signed [INT32_WIDTH-1:0] matrix_d_10, matrix_d_11, matrix_d_12, matrix_d_13,
    output reg signed [INT32_WIDTH-1:0] matrix_d_20, matrix_d_21, matrix_d_22, matrix_d_23,
    output reg signed [INT32_WIDTH-1:0] matrix_d_30, matrix_d_31, matrix_d_32, matrix_d_33
);

    // Pack inputs into arrays for easier handling
    reg signed [INT8_WIDTH-1:0] a [0:3][0:3];
    reg signed [INT8_WIDTH-1:0] b [0:3][0:3];
    reg signed [INT32_WIDTH-1:0] c [0:3][0:3];
    reg signed [INT32_WIDTH-1:0] d [0:3][0:3];
    
    always @(*) begin
        // Pack A matrix
        a[0][0] = matrix_a_00; a[0][1] = matrix_a_01; a[0][2] = matrix_a_02; a[0][3] = matrix_a_03;
        a[1][0] = matrix_a_10; a[1][1] = matrix_a_11; a[1][2] = matrix_a_12; a[1][3] = matrix_a_13;
        a[2][0] = matrix_a_20; a[2][1] = matrix_a_21; a[2][2] = matrix_a_22; a[2][3] = matrix_a_23;
        a[3][0] = matrix_a_30; a[3][1] = matrix_a_31; a[3][2] = matrix_a_32; a[3][3] = matrix_a_33;
        
        // Pack B matrix
        b[0][0] = matrix_b_00; b[0][1] = matrix_b_01; b[0][2] = matrix_b_02; b[0][3] = matrix_b_03;
        b[1][0] = matrix_b_10; b[1][1] = matrix_b_11; b[1][2] = matrix_b_12; b[1][3] = matrix_b_13;
        b[2][0] = matrix_b_20; b[2][1] = matrix_b_21; b[2][2] = matrix_b_22; b[2][3] = matrix_b_23;
        b[3][0] = matrix_b_30; b[3][1] = matrix_b_31; b[3][2] = matrix_b_32; b[3][3] = matrix_b_33;
        
        // Pack C matrix
        c[0][0] = matrix_c_00; c[0][1] = matrix_c_01; c[0][2] = matrix_c_02; c[0][3] = matrix_c_03;
        c[1][0] = matrix_c_10; c[1][1] = matrix_c_11; c[1][2] = matrix_c_12; c[1][3] = matrix_c_13;
        c[2][0] = matrix_c_20; c[2][1] = matrix_c_21; c[2][2] = matrix_c_22; c[2][3] = matrix_c_23;
        c[3][0] = matrix_c_30; c[3][1] = matrix_c_31; c[3][2] = matrix_c_32; c[3][3] = matrix_c_33;
    end
    
    // State machine
    localparam IDLE = 2'b00;
    localparam COMPUTING = 2'b01;
    localparam DONE = 2'b10;
    
    reg [1:0] state;
    
    // Intermediate results for matrix multiplication
    reg signed [31:0] result [0:3][0:3];
    
    // Compute D = A × B + C
    integer i, j, k;
    always @(*) begin
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                result[i][j] = c[i][j];
                for (k = 0; k < 4; k = k + 1) begin
                    result[i][j] = result[i][j] + (a[i][k] * b[k][j]);
                end
            end
        end
    end
    
    // State machine and output logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            result_valid <= 1'b0;
            mma_ready <= 1'b1;
            d[0][0] <= 0; d[0][1] <= 0; d[0][2] <= 0; d[0][3] <= 0;
            d[1][0] <= 0; d[1][1] <= 0; d[1][2] <= 0; d[1][3] <= 0;
            d[2][0] <= 0; d[2][1] <= 0; d[2][2] <= 0; d[2][3] <= 0;
            d[3][0] <= 0; d[3][1] <= 0; d[3][2] <= 0; d[3][3] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    mma_ready <= 1'b1;
                    result_valid <= 1'b0;
                    if (mma_valid && mma_enable) begin
                        state <= COMPUTING;
                        mma_ready <= 1'b0;
                    end
                end
                
                COMPUTING: begin
                    // Latch results
                    d[0][0] <= result[0][0]; d[0][1] <= result[0][1]; 
                    d[0][2] <= result[0][2]; d[0][3] <= result[0][3];
                    d[1][0] <= result[1][0]; d[1][1] <= result[1][1]; 
                    d[1][2] <= result[1][2]; d[1][3] <= result[1][3];
                    d[2][0] <= result[2][0]; d[2][1] <= result[2][1]; 
                    d[2][2] <= result[2][2]; d[2][3] <= result[2][3];
                    d[3][0] <= result[3][0]; d[3][1] <= result[3][1]; 
                    d[3][2] <= result[3][2]; d[3][3] <= result[3][3];
                    
                    result_valid <= 1'b1;
                    state <= DONE;
                end
                
                DONE: begin
                    result_valid <= 1'b0;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end
    
    // Unpack D matrix to output ports
    always @(*) begin
        matrix_d_00 = d[0][0]; matrix_d_01 = d[0][1]; matrix_d_02 = d[0][2]; matrix_d_03 = d[0][3];
        matrix_d_10 = d[1][0]; matrix_d_11 = d[1][1]; matrix_d_12 = d[1][2]; matrix_d_13 = d[1][3];
        matrix_d_20 = d[2][0]; matrix_d_21 = d[2][1]; matrix_d_22 = d[2][2]; matrix_d_23 = d[2][3];
        matrix_d_30 = d[3][0]; matrix_d_31 = d[3][1]; matrix_d_32 = d[3][2]; matrix_d_33 = d[3][3];
    end

endmodule


// Testbench
module tb_tensor_core;
    
    parameter CLK_PERIOD = 10;
    
    reg clk, rst_n;
    reg mma_enable, mma_valid;
    wire mma_ready, result_valid;
    
    // Input matrices
    reg signed [7:0] a_00, a_01, a_02, a_03;
    reg signed [7:0] a_10, a_11, a_12, a_13;
    reg signed [7:0] a_20, a_21, a_22, a_23;
    reg signed [7:0] a_30, a_31, a_32, a_33;
    
    reg signed [7:0] b_00, b_01, b_02, b_03;
    reg signed [7:0] b_10, b_11, b_12, b_13;
    reg signed [7:0] b_20, b_21, b_22, b_23;
    reg signed [7:0] b_30, b_31, b_32, b_33;
    
    reg signed [31:0] c_00, c_01, c_02, c_03;
    reg signed [31:0] c_10, c_11, c_12, c_13;
    reg signed [31:0] c_20, c_21, c_22, c_23;
    reg signed [31:0] c_30, c_31, c_32, c_33;
    
    // Output matrix
    wire signed [31:0] d_00, d_01, d_02, d_03;
    wire signed [31:0] d_10, d_11, d_12, d_13;
    wire signed [31:0] d_20, d_21, d_22, d_23;
    wire signed [31:0] d_30, d_31, d_32, d_33;
    
    // DUT
    tensor_core_int8 dut (
        .clk(clk),
        .rst_n(rst_n),
        .mma_enable(mma_enable),
        .mma_valid(mma_valid),
        .mma_ready(mma_ready),
        .result_valid(result_valid),
        
        .matrix_a_00(a_00), .matrix_a_01(a_01), .matrix_a_02(a_02), .matrix_a_03(a_03),
        .matrix_a_10(a_10), .matrix_a_11(a_11), .matrix_a_12(a_12), .matrix_a_13(a_13),
        .matrix_a_20(a_20), .matrix_a_21(a_21), .matrix_a_22(a_22), .matrix_a_23(a_23),
        .matrix_a_30(a_30), .matrix_a_31(a_31), .matrix_a_32(a_32), .matrix_a_33(a_33),
        
        .matrix_b_00(b_00), .matrix_b_01(b_01), .matrix_b_02(b_02), .matrix_b_03(b_03),
        .matrix_b_10(b_10), .matrix_b_11(b_11), .matrix_b_12(b_12), .matrix_b_13(b_13),
        .matrix_b_20(b_20), .matrix_b_21(b_21), .matrix_b_22(b_22), .matrix_b_23(b_23),
        .matrix_b_30(b_30), .matrix_b_31(b_31), .matrix_b_32(b_32), .matrix_b_33(b_33),
        
        .matrix_c_00(c_00), .matrix_c_01(c_01), .matrix_c_02(c_02), .matrix_c_03(c_03),
        .matrix_c_10(c_10), .matrix_c_11(c_11), .matrix_c_12(c_12), .matrix_c_13(c_13),
        .matrix_c_20(c_20), .matrix_c_21(c_21), .matrix_c_22(c_22), .matrix_c_23(c_23),
        .matrix_c_30(c_30), .matrix_c_31(c_31), .matrix_c_32(c_32), .matrix_c_33(c_33),
        
        .matrix_d_00(d_00), .matrix_d_01(d_01), .matrix_d_02(d_02), .matrix_d_03(d_03),
        .matrix_d_10(d_10), .matrix_d_11(d_11), .matrix_d_12(d_12), .matrix_d_13(d_13),
        .matrix_d_20(d_20), .matrix_d_21(d_21), .matrix_d_22(d_22), .matrix_d_23(d_23),
        .matrix_d_30(d_30), .matrix_d_31(d_31), .matrix_d_32(d_32), .matrix_d_33(d_33)
    );
    
    // Clock
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Test
    initial begin
        $display("=== NVIDIA Tensor Core Testbench ===");
        $display("Computing: D = A x B + C");
        $display("4x4 INT8 matrices\n");
        
        // Initialize
        rst_n = 0;
        mma_enable = 1;
        mma_valid = 0;
        
        // Matrix A: Near identity (2 on diagonal, 1 elsewhere)
        a_00 = 2; a_01 = 1; a_02 = 1; a_03 = 1;
        a_10 = 1; a_11 = 2; a_12 = 1; a_13 = 1;
        a_20 = 1; a_21 = 1; a_22 = 2; a_23 = 1;
        a_30 = 1; a_31 = 1; a_32 = 1; a_33 = 2;
        
        // Matrix B: All ones
        b_00 = 1; b_01 = 1; b_02 = 1; b_03 = 1;
        b_10 = 1; b_11 = 1; b_12 = 1; b_13 = 1;
        b_20 = 1; b_21 = 1; b_22 = 1; b_23 = 1;
        b_30 = 1; b_31 = 1; b_32 = 1; b_33 = 1;
        
        // Matrix C: Zero (no accumulation)
        c_00 = 3; c_01 = 0; c_02 = 0; c_03 = 0;
        c_10 = 0; c_11 = 0; c_12 = 0; c_13 = 0;
        c_20 = 2; c_21 = 0; c_22 = 5; c_23 = 0;
        c_30 = 0; c_31 = 0; c_32 = 0; c_33 = 0;
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        $display("Matrix A (INT8):");
        $display("  [ %3d %3d %3d %3d ]", a_00, a_01, a_02, a_03);
        $display("  [ %3d %3d %3d %3d ]", a_10, a_11, a_12, a_13);
        $display("  [ %3d %3d %3d %3d ]", a_20, a_21, a_22, a_23);
        $display("  [ %3d %3d %3d %3d ]", a_30, a_31, a_32, a_33);
        
        $display("\nMatrix B (INT8):");
        $display("  [ %3d %3d %3d %3d ]", b_00, b_01, b_02, b_03);
        $display("  [ %3d %3d %3d %3d ]", b_10, b_11, b_12, b_13);
        $display("  [ %3d %3d %3d %3d ]", b_20, b_21, b_22, b_23);
        $display("  [ %3d %3d %3d %3d ]", b_30, b_31, b_32, b_33);
        
        // Start computation
        $display("\nStarting MMA operation...");
        @(posedge clk);
        mma_valid = 1;
        @(posedge clk);
        mma_valid = 0;
        
        // Wait for result
        @(posedge result_valid);
        @(posedge clk);
        
        $display("\nMatrix D = A x B + C (INT32):");
        $display("  [ %3d %3d %3d %3d ]", d_00, d_01, d_02, d_03);
        $display("  [ %3d %3d %3d %3d ]", d_10, d_11, d_12, d_13);
        $display("  [ %3d %3d %3d %3d ]", d_20, d_21, d_22, d_23);
        $display("  [ %3d %3d %3d %3d ]", d_30, d_31, d_32, d_33);
        
        $display("\nExpected: Each row sums to 5 (2+1+1+1)");
        $display("Tensor Core computes in 1-2 cycles!");
        
        repeat(5) @(posedge clk);
        $display("\n=== Test Complete ===");
        $finish;
    end
    
    // Timeout
    initial begin
        #10000;
        $display("ERROR: Timeout");
        $finish;
    end

endmodule
