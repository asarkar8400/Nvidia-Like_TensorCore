module memory #(   
        parameter                   WIDTH=16, SIZE=64,
        localparam                  LOGSIZE=$clog2(SIZE)
    )(
        input  [WIDTH-1:0]          data_in,
        output logic [WIDTH-1:0]    data_out,
        input  [LOGSIZE-1:0]        addr,
        input                       clk, wr_en
    );

    logic [SIZE-1:0][WIDTH-1:0] mem;
    
    always_ff @(posedge clk) begin
        data_out <= mem[addr];
        if (wr_en)
            mem[addr] <= data_in;
    end
endmodule

module input_mems #(
    parameter INW  = 24,
    parameter R    = 9,
    parameter C    = 8,
    parameter MAXK = 4,
    localparam K_BITS      = $clog2(MAXK+1),
    localparam X_ADDR_BITS = $clog2(R*C),
    localparam W_ADDR_BITS = $clog2(MAXK*MAXK)
)(
    input                    clk,
    input                    reset,

    // AXI-Stream input side
    input      [INW-1:0]     AXIS_TDATA,
    input                    AXIS_TVALID,
    input      [K_BITS:0]    AXIS_TUSER,
    output logic             AXIS_TREADY,

    // Status / control to rest of system
    output logic             inputs_loaded,
    input                    compute_finished,
    output logic [K_BITS-1:0]     K,
    output logic signed [INW-1:0] B,

    // Read interfaces for X and W memories
    input      [X_ADDR_BITS-1:0]  X_read_addr,
    output logic signed [INW-1:0] X_data,
    input      [W_ADDR_BITS-1:0]  W_read_addr,
    output logic signed [INW-1:0] W_data
);

    // -----------------------------------------------------------------
    // Decode AXIS_TUSER into TUSER_K and new_W (spec Figure 3.3/3.4)
    // -----------------------------------------------------------------
    logic [K_BITS-1:0] TUSER_K;
    assign TUSER_K = AXIS_TUSER[K_BITS:1];

    logic new_W;
    assign new_W  = AXIS_TUSER[0];

    // -----------------------------------------------------------------
    // Internal memories for W and X
    // -----------------------------------------------------------------
    logic [INW-1:0] w_data_out, x_data_out;
    logic [W_ADDR_BITS-1:0] w_addr;
    logic [X_ADDR_BITS-1:0] x_addr;
    logic w_wr_en, x_wr_en;

    memory #(.WIDTH(INW), .SIZE(MAXK*MAXK)) w_memory (
        .clk    (clk),
        .data_in(AXIS_TDATA),
        .data_out(w_data_out),
        .addr   (w_addr),
        .wr_en  (w_wr_en)
    );

    memory #(.WIDTH(INW), .SIZE(R*C)) x_memory (
        .clk    (clk),
        .data_in(AXIS_TDATA),
        .data_out(x_data_out),
        .addr   (x_addr),
        .wr_en  (x_wr_en)
    );

    assign W_data = w_data_out;
    assign X_data = x_data_out;

    // -----------------------------------------------------------------
    // State machine
    // -----------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,        // waiting for first input of a new operation
        S_LOAD_W,      // loading W matrix
        S_LOAD_B,      // loading bias B
        S_LOAD_X,      // loading X matrix
        S_INPUTS_LOADED// inputs ready, waiting for compute_finished
    } state_t;

    state_t state, state_next;

    // Registers for K and B (outputs)
    logic [K_BITS-1:0]      K_reg, K_next;
    logic signed [INW-1:0]  B_reg, B_next;

    // Counters / indices for addressing memories while loading
    logic [W_ADDR_BITS-1:0] w_index_reg, w_index_next;
    logic [X_ADDR_BITS-1:0] x_index_reg, x_index_next;

    // Handy constants
    localparam int X_LAST = R*C - 1;

    // Small helper: number of W entries to load is K*K
    logic [W_ADDR_BITS-1:0] last_w_index;
    always_comb begin
        // K in [2, MAXK]; K*K <= MAXK*MAXK <= 16 here,
        // fits in W_ADDR_BITS bits. We subtract 1 to get last index.
        last_w_index = (K_reg * K_reg) - 1;
    end

    // AXI-Stream handshake
    logic input_fire;
    assign input_fire = AXIS_TVALID && AXIS_TREADY;

    // -----------------------------------------------------------------
    // Combinational FSM + control logic
    // -----------------------------------------------------------------
    always_comb begin
        // defaults
        state_next      = state;
        K_next          = K_reg;
        B_next          = B_reg;
        w_index_next    = w_index_reg;
        x_index_next    = x_index_reg;

        AXIS_TREADY     = 1'b0;
        inputs_loaded   = 1'b0;

        w_wr_en         = 1'b0;
        x_wr_en         = 1'b0;

        // Default memory addresses:
        // - When inputs_loaded==1: driven by external read addresses
        // - Otherwise: driven by internal load indices
        w_addr = (state == S_INPUTS_LOADED) ? W_read_addr : w_index_reg;
        x_addr = (state == S_INPUTS_LOADED) ? X_read_addr : x_index_reg;

        case (state)
            // ---------------------------------------------------------
            // S_IDLE: First valid word of a new operation.
            // If new_W==1: load W (and K) first.
            // If new_W==0: reuse old W,B,K and directly load X.
            // ---------------------------------------------------------
            S_IDLE: begin
                AXIS_TREADY = 1'b1;
                if (input_fire) begin
                    if (new_W) begin
                        // First word is W[0][0], and TUSER_K is K.
                        K_next       = TUSER_K;
                        // write W[0]
                        w_wr_en      = 1'b1;
                        w_addr       = '0;
                        w_index_next = 1;   // next W index
                        x_index_next = '0;
                        state_next   = S_LOAD_W;
                    end
                    else begin
                        // Reuse old W and B; this is X[0][0]
                        x_wr_en      = 1'b1;
                        x_addr       = '0;
                        x_index_next = 1;
                        // K_reg and B_reg remain unchanged
                        state_next   = S_LOAD_X;
                    end
                end
            end

            // ---------------------------------------------------------
            // S_LOAD_W: Loading remaining W entries (we already wrote
            // W[0] in S_IDLE when new_W==1).
            // ---------------------------------------------------------
            S_LOAD_W: begin
                AXIS_TREADY = 1'b1;
                if (input_fire) begin
                    w_wr_en = 1'b1;
                    w_addr  = w_index_reg;

                    if (w_index_reg == last_w_index) begin
                        // Completed K*K entries
                        w_index_next = '0;
                        state_next   = S_LOAD_B;
                    end
                    else begin
                        w_index_next = w_index_reg + 1;
                    end
                end
            end

            // ---------------------------------------------------------
            // S_LOAD_B: Single bias word B
            // ---------------------------------------------------------
            S_LOAD_B: begin
                AXIS_TREADY = 1'b1;
                if (input_fire) begin
                    B_next        = AXIS_TDATA;
                    x_index_next  = '0;    // start X at address 0
                    state_next    = S_LOAD_X;
                end
            end

            // ---------------------------------------------------------
            // S_LOAD_X: Load R*C entries of X
            // ---------------------------------------------------------
            S_LOAD_X: begin
                AXIS_TREADY = 1'b1;
                if (input_fire) begin
                    x_wr_en = 1'b1;
                    x_addr  = x_index_reg;

                    if (x_index_reg == X_LAST[X_ADDR_BITS-1:0]) begin
                        x_index_next  = '0;
                        state_next    = S_INPUTS_LOADED;
                    end
                    else begin
                        x_index_next = x_index_reg + 1;
                    end
                end
            end

            // ---------------------------------------------------------
            // S_INPUTS_LOADED: Inputs are ready.
            // - AXIS_TREADY must be 0.
            // - K and B must be stable/valid.
            // - Allow external logic to read memories.
            // - Wait for compute_finished to go back to S_IDLE.
            // ---------------------------------------------------------
            S_INPUTS_LOADED: begin
                AXIS_TREADY   = 1'b0;
                inputs_loaded = 1'b1;

                // Memory addresses already muxed to X_read_addr/W_read_addr
                if (compute_finished) begin
                    state_next   = S_IDLE;
                    w_index_next = '0;
                    x_index_next = '0;
                    // Keep K_reg and B_reg so they can be reused if new_W==0.
                end
            end

            default: begin
                state_next = S_IDLE;
            end
        endcase
    end

    // -----------------------------------------------------------------
    // Sequential logic
    // -----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            state       <= S_IDLE;
            K_reg       <= '0;
            B_reg       <= '0;
            w_index_reg <= '0;
            x_index_reg <= '0;
        end
        else begin
            state       <= state_next;
            K_reg       <= K_next;
            B_reg       <= B_next;
            w_index_reg <= w_index_next;
            x_index_reg <= x_index_next;
        end
    end

    // Drive outputs
    assign K = K_reg;
    assign B = B_reg;

endmodule
